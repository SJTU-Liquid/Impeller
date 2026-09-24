use group::ff::PrimeField;
use halo2curves::bn256::Fr;
use zkpoly_compiler::test_util::{prelude::*, Tester};

mod common;
use common::RtInstance;

fn config() -> driver::Config {
    driver::Config::default().with_sliceable_subgraph_on(
        driver::SubgraphSlicingConfig::default()
            .with_chunk_len(16)
            .with_minimum_order(2),
    )
}

#[test]
fn test_one_small() {
    let deg = 1024;

    let ast = {
        let mut entry_definer = ast::EntryDefiner::new();

        let a: ast::PolyLagrange<RtInstance> =
            entry_definer.define("a".to_string(), Typ::lagrange(deg));
        let b: ast::PolyLagrange<RtInstance> =
            entry_definer.define("b".to_string(), Typ::lagrange(deg));

        a * b.invert()
    };

    let mut tester = Tester::small();

    let deg = deg as usize;

    let mut a = tester.scalar_array::<RtInstance>(deg);
    let mut b = tester.scalar_array::<RtInstance>(deg);

    for (i, x) in a.iter_mut().enumerate() {
        *x = Fr::from_u128((i as u128).pow(2));
    }

    for (i, x) in b.iter_mut().enumerate() {
        *x = Fr::from_u128(i as u128);
    }

    let mut inputs = Heap::new();
    inputs.push(Variable::ScalarArray(a));
    inputs.push(Variable::ScalarArray(b));

    let (r, _pool) = tester.run(
        ast,
        &config(),
        &inputs,
        "../target/tests/test_one_small_sliceable_subgraph",
    );

    let mut answer = tester.scalar_array::<RtInstance>(deg);
    for (i, x) in answer.iter_mut().enumerate() {
        *x = Fr::from_u128(i as u128);
    }

    assert_eq!(r.unwrap().unwrap_scalar_array(), &answer);
}

#[test]
fn test_small_rotated() {
    let deg = 128;
    let rot = 4;

    let ast = {
        let mut entry_definer = ast::EntryDefiner::new();

        let a: ast::PolyLagrange<RtInstance> =
            entry_definer.define("a".to_string(), Typ::lagrange(deg));
        let b: ast::PolyLagrange<RtInstance> =
            entry_definer.define("b".to_string(), Typ::lagrange(deg));

        a * b.rotate(rot)
    };

    let mut tester = Tester::small();

    let deg = deg as usize;

    let mut a = tester.scalar_array::<RtInstance>(deg);
    let mut b = tester.scalar_array::<RtInstance>(deg);

    for (i, x) in a.iter_mut().enumerate() {
        *x = Fr::from_u128(i as u128);
    }

    for (i, x) in b.iter_mut().enumerate() {
        *x = Fr::from_u128(i as u128);
    }

    let mut answer = tester.scalar_array::<RtInstance>(deg);
    for (i, x) in answer.iter_mut().enumerate() {
        let b_idx = (i as i64 + rot as i64).rem_euclid(deg as i64);
        *x = a[i] * b[b_idx as usize];
    }

    let mut inputs = Heap::new();
    inputs.push(Variable::ScalarArray(a));
    inputs.push(Variable::ScalarArray(b));

    let (r, _pool) = tester.run(
        ast,
        &config(),
        &inputs,
        "../target/tests/test_small_rotated_sliceable_subgraph",
    );

    assert_eq!(r.unwrap().unwrap_scalar_array(), &answer);
}

#[test]
fn test_successive_rotated() {
    let deg = 128;
    let rot = 33;

    let ast = {
        let mut entry_definer = ast::EntryDefiner::new();

        let a: ast::PolyLagrange<RtInstance> =
            entry_definer.define("a".to_string(), Typ::lagrange(deg));
        let b: ast::PolyLagrange<RtInstance> =
            entry_definer.define("b".to_string(), Typ::lagrange(deg));

        let c = a.clone() * b.rotate(rot) + a.rotate(rot);
        let r = b * c.rotate(rot) * c;
        r
    };

    let mut tester = Tester::small();

    let deg = deg as usize;

    let mut a = tester.scalar_array::<RtInstance>(deg);
    let mut b = tester.scalar_array::<RtInstance>(deg);

    for (i, x) in a.iter_mut().enumerate() {
        *x = Fr::from_u128(i as u128);
    }

    for (_i, x) in b.iter_mut().enumerate() {
        *x = Fr::from_u128(1 as u128);
    }

    let mut answer = tester.scalar_array::<RtInstance>(deg);
    for (i, x) in answer.iter_mut().enumerate() {
        let rot_idx = (i as i64 + rot as i64).rem_euclid(deg as i64) as usize;
        let rot2_idx = (i as i64 + 2 * rot as i64).rem_euclid(deg as i64) as usize;
        *x = b[i] * (a[rot_idx] * b[rot2_idx] + a[rot2_idx]) * (a[i] * b[rot_idx] + a[rot_idx]);
    }

    let mut inputs = Heap::new();
    inputs.push(Variable::ScalarArray(a));
    inputs.push(Variable::ScalarArray(b));

    let (r, _pool) = tester.run(
        ast,
        &config(),
        &inputs,
        "../target/tests/test_successive_rotated_sliceable_subgraph",
    );

    assert_eq!(
        r.unwrap().unwrap_scalar_array(),
        &answer,
        "equality failed at rotation {rot}"
    );
}

#[test]
fn test_successive_negative_rotated_eight_chunks() {
    test_successive_negative_rotated_chunks(
        128,
        "../target/tests/test_successive_negative_rotated_eight_chunks",
    );
}

#[test]
fn test_successive_negative_rotated_sixteen_chunks() {
    test_successive_negative_rotated_chunks(
        256,
        "../target/tests/test_successive_negative_rotated_sixteen_chunks",
    );
}

fn test_successive_negative_rotated_chunks(deg: u64, test_dir: &str) {
    let rot = -33;

    let ast = {
        let mut entry_definer = ast::EntryDefiner::new();

        let a: ast::PolyLagrange<RtInstance> =
            entry_definer.define("a".to_string(), Typ::lagrange(deg));
        let b: ast::PolyLagrange<RtInstance> =
            entry_definer.define("b".to_string(), Typ::lagrange(deg));

        let c = a.clone() * b.rotate(rot) + a.rotate(rot);
        b.clone() * c.rotate(rot) * c
    };

    // The default 32 MiB smithereen pool leaves 16 KiB of integral GPU
    // space. For this graph, chunk tightening therefore keeps the requested
    // chunk length of 16.
    let mut tester = Tester::small_with_gpu_memory_limit(2u64.pow(25) + 2u64.pow(14));
    let deg = deg as usize;

    let mut a = tester.scalar_array::<RtInstance>(deg);
    let mut b = tester.scalar_array::<RtInstance>(deg);

    for (i, x) in a.iter_mut().enumerate() {
        *x = Fr::from_u128(i as u128);
    }

    for x in b.iter_mut() {
        *x = Fr::from_u128(1);
    }

    let mut answer = tester.scalar_array::<RtInstance>(deg);
    for (i, x) in answer.iter_mut().enumerate() {
        let rot_idx = (i as i64 + rot as i64).rem_euclid(deg as i64) as usize;
        let rot2_idx = (i as i64 + 2 * rot as i64).rem_euclid(deg as i64) as usize;
        *x = b[i] * (a[rot_idx] * b[rot2_idx] + a[rot2_idx]) * (a[i] * b[rot_idx] + a[rot_idx]);
    }

    let mut inputs = Heap::new();
    inputs.push(Variable::ScalarArray(a));
    inputs.push(Variable::ScalarArray(b));

    let (r, _pool) = tester.run(
        ast,
        &config(),
        &inputs,
        test_dir,
    );

    assert_eq!(
        r.unwrap().unwrap_scalar_array(),
        &answer,
        "equality failed at rotation {rot}"
    );
}
