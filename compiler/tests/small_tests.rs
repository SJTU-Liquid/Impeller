use group::ff::PrimeField;
use halo2curves::bn256::Fr;
use zkpoly_compiler::test_util::{prelude::*, Tester};

mod common;
use common::RtInstance;

#[test]
fn test_extend() {
    let ast = {
        let mut entry_definer = ast::EntryDefiner::new();

        let input: ast::PolyLagrange<RtInstance> =
            entry_definer.define("input".to_string(), Typ::lagrange(128));
        // let input = input.to_coef();
        let extended = input.extend(256);
        extended
    };

    let mut tester = Tester::small();
    let mut arr0 = tester.scalar_array::<RtInstance>(128);

    for (i, x) in arr0.iter_mut().enumerate() {
        *x = Fr::from_u128(i.try_into().unwrap())
    }

    let mut answer = tester.scalar_array::<RtInstance>(256);
    for (i, x) in answer.iter_mut().enumerate() {
        *x = Fr::from_u128(i.try_into().unwrap())
    }
    for x in answer.iter_mut().skip(128) {
        *x = Fr::zero();
    }

    let mut inputs = Heap::new();
    inputs.push(Variable::ScalarArray(arr0));
    let (r, _pool) = tester.run(
        ast,
        &driver::Config::default(),
        &inputs,
        "../target/tests/test_extend",
    );
    assert_eq!(&answer, r.unwrap().unwrap_scalar_array());
}
