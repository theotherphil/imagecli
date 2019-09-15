//! Support for rendering markdown diagrams showing the evolution of the
//! image stack as a pipeline is evaluated.

use std::fmt::Write;

/// A diagram showing the evolution of the image stack
/// as a pipeline is evaluated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StackDiagram {
    /// The stages to render, in order.
    pub stages: Vec<StackDiagramStage>,
}

/// A stage in a stack diagram.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StackDiagramStage {
    /// A desription of what occurs between one stack state and the next.
    Transition(String),
    /// A list of the current stack contents.
    Stack(Vec<String>),
}

impl StackDiagram {
    /// Render the stack in a form suitable for inclusion in a markdown document.
    pub fn render_for_markdown(&self) -> String {
        let mut rendered = String::new();
        writeln!(rendered, "```").unwrap();
        for stage in &self.stages {
            match stage {
                StackDiagramStage::Transition(t) => {
                    writeln!(rendered, "    |").unwrap();
                    writeln!(rendered, "    | {}", t).unwrap();
                    writeln!(rendered, "    v").unwrap();
                }
                StackDiagramStage::Stack(s) => {
                    writeln!(rendered, " --------------------").unwrap();
                    let mut top = true;
                    for entry in s.iter() {
                        if top {
                            writeln!(rendered, " * {}", entry).unwrap();
                        } else {
                            writeln!(rendered, "   {}", entry).unwrap();
                        }
                        top = false;
                    }
                    writeln!(rendered, " --------------------").unwrap();
                }
            }
        }
        writeln!(rendered, "```").unwrap();
        rendered
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_render_stack_diagram() {
        let diagram = StackDiagram {
            stages: vec![
                StackDiagramStage::Transition("get inputs".into()),
                StackDiagramStage::Stack(vec!["A".into(), "B".into()]),
                StackDiagramStage::Transition("some operation".into()),
                StackDiagramStage::Stack(vec!["C".into(), "B".into()]),
                StackDiagramStage::Transition("save results".into()),
            ],
        };
        let expected = "```
    |
    | get inputs
    v
 --------------------
 * A
   B
 --------------------
    |
    | some operation
    v
 --------------------
 * C
   B
 --------------------
    |
    | save results
    v
```
";
        assert_eq!(diagram.render_for_markdown(), expected);
    }
}
