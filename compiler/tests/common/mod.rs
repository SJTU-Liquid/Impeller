use group::ff::PrimeField;
use halo2curves::bn256::Fr;
use zkpoly_compiler::test_util::{prelude::*, Tester};
use zkpoly_runtime::transcript::{Blake2bWrite, Challenge255};

#[derive(Clone, Hash, Debug)]
pub struct RtInstance;

impl RuntimeType for RtInstance {
    type Field = halo2curves::bn256::Fr;
    type PointAffine = halo2curves::bn256::G1Affine;
    type Challenge = Challenge255<halo2curves::bn256::G1Affine>;
    type Trans = Blake2bWrite<Vec<u8>, halo2curves::bn256::G1Affine, Self::Challenge>;
}
