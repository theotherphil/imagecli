
use crate::ImageStack;

/// Direct manipulations of the image stack.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum StackOp {
    Dup(usize),
    Drop,
    Over,
    Rot(usize)
}

impl StackOp {
    pub fn parse(op: &str) -> Option<Self> {
        let split: Vec<&str> = op.split_whitespace().collect();
        match split[0] {
            "DUP" => {
                if split.len() == 1 {
                    Some(StackOp::Dup(1))
                } else {
                    Some(StackOp::Dup(split[1].parse().unwrap()))
                }
            },
            "DROP" => Some(StackOp::Drop),
            "OVER" => Some(StackOp::Over),
            "SWAP" => Some(StackOp::Rot(2)),
            "ROT" => {
                if split.len() == 1 {
                    Some(StackOp::Rot(3))
                } else {
                    Some(StackOp::Rot(split[1].parse().unwrap()))
                }
            },
            _ => None,
        }
    }

    pub fn apply(&self, stack: &mut ImageStack) {
        // TODO: inline stack function definitions here?
        match self {
            StackOp::Dup(n) => stack.dup(*n),
            StackOp::Drop => stack.drop(),
            StackOp::Over => stack.over(),
            StackOp::Rot(n) => stack.rot(*n),
        }
    }
}
