use crate::args::{RuntimeType, Variable};

pub fn assert_eq<Rt: RuntimeType>(value: &Variable<Rt>, expected: &Variable<Rt>) -> bool {
    match (value, expected) {
        (Variable::Scalar(a), Variable::Scalar(b)) => {
            assert!(a.device.is_cpu());
            assert!(b.device.is_cpu());
            *a.as_ref() == *b.as_ref()
        }
        (Variable::Any(_), Variable::Any(_)) => unreachable!("any can't be compared"),
        (Variable::GpuBuffer(_), Variable::GpuBuffer(_)) => {
            unreachable!("gpu buffer can't be compared")
        }
        (Variable::Point(a), Variable::Point(b)) => *a.as_ref() == *b.as_ref(),
        (Variable::PointArray(a), Variable::PointArray(b)) => {
            assert!(a.device.is_cpu());
            assert!(b.device.is_cpu());
            a == b
        }
        (Variable::ScalarArray(a), Variable::ScalarArray(b)) => {
            assert!(a.device.is_cpu());
            assert!(b.device.is_cpu());
            a == b
        }
        (Variable::Stream(_), Variable::Stream(_)) => unreachable!("stream can't be compared"),
        (Variable::Tuple(a), Variable::Tuple(b)) => {
            if a.len() != b.len() {
                return false;
            }
            for (a, b) in a.iter().zip(b.iter()) {
                if !assert_eq(a, b) {
                    return false;
                }
            }
            true
        }
        (Variable::Transcript(_), Variable::Transcript(_)) => {
            unimplemented!("transcript doesn't implement PartialEq")
        }
        _ => false,
    }
}

pub fn diagnose<Rt: RuntimeType>(value: &Variable<Rt>, expected: &Variable<Rt>) -> String {
    match (value, expected) {
        (Variable::Scalar(a), Variable::Scalar(b)) => {
            format!("scalar value {:?}, expected {:?}", a.as_ref(), b.as_ref())
        }
        (Variable::Point(a), Variable::Point(b)) => {
            format!("point value {:?}, expected {:?}", a.as_ref(), b.as_ref())
        }
        (Variable::PointArray(a), Variable::PointArray(b)) => {
            if a.len != b.len {
                return format!("point array length {}, expected {}", a.len, b.len);
            }
            match (0..a.len).find(|&i| a[i] != b[i]) {
                Some(i) => format!("point array first differs at logical index {i}"),
                None => format!(
                    "point array metadata differs: device {:?}, expected {:?}",
                    a.device, b.device
                ),
            }
        }
        (Variable::ScalarArray(a), Variable::ScalarArray(b)) => {
            if a.len != b.len {
                return format!("scalar array length {}, expected {}", a.len, b.len);
            }
            match (0..a.len).find(|&i| a[i] != b[i]) {
                Some(i) => format!(
                    "scalar array first differs at logical index {i}; \
                     value metadata: rotation {}, slice {:?}; \
                     expected metadata: rotation {}, slice {:?}",
                    a.get_rotation(),
                    a.slice_info,
                    b.get_rotation(),
                    b.slice_info
                ),
                None => "scalar array values are equal but metadata differs".to_string(),
            }
        }
        (Variable::Tuple(a), Variable::Tuple(b)) => {
            if a.len() != b.len() {
                return format!("tuple length {}, expected {}", a.len(), b.len());
            }
            match a
                .iter()
                .zip(b.iter())
                .enumerate()
                .find(|(_, (a, b))| !assert_eq(a, b))
            {
                Some((i, (a, b))) => format!("tuple element {i}: {}", diagnose(a, b)),
                None => "tuple values are equal".to_string(),
            }
        }
        (a, b) => format!(
            "variable variants differ: value {:?}, expected {:?}",
            std::mem::discriminant(a),
            std::mem::discriminant(b)
        ),
    }
}
