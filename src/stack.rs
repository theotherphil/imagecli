
// TODO: Write tests for all Stack functions.

pub struct Stack<T> {
    // The top of the stack is the last element in the vector.
    contents: Vec<T>,
}

impl<T: Clone> Stack<T> {
    /// Create a stack with the given initial elements.
    /// The top of the stack is the last item in the provided vector.
    pub fn new(contents: Vec<T>) -> Self {
        Stack { contents }
    }

    /// dup ( a -- a a )
    pub fn dup(&mut self) {
        self.assert_stack_size("dup", 1);
        self.contents.push(self.contents[self.len() - 1].clone());
    }

    /// drop ( a -- )
    pub fn drop(&mut self) {
        self.assert_stack_size("drop", 1);
        self.contents.remove(self.len() - 1);
    }

    /// swap ( a b -- b a )
    pub fn swap(&mut self) {
        self.assert_stack_size("swap", 2);
        let (i, j) = (self.len() - 1, self.len() - 2);
        self.contents.swap(i, j);
    }

    /// over ( a b -- a b a )
    pub fn over(&mut self) {
        self.assert_stack_size("over", 2);
        let a = self.contents[self.len() - 1].clone();
        self.contents.insert(self.len() - 2, a);
    }

    /// rot ( a b c -- b c a )
    pub fn rot(&mut self) {
        self.assert_stack_size("rot", 3);
        let a = self.contents[self.len() - 1].clone();
        self.contents.remove(self.contents.len() - 1);
        self.contents.insert(self.len() - 3, a);
    }

    /// pops the top of the stack.
    pub fn pop(&mut self) -> T {
        assert!(!self.contents.is_empty(), "cannot pop an empty stack");
        self.contents.pop().unwrap()
    }

    /// pushes onto the top of the stack.
    pub fn push(&mut self, x: T) {
        self.contents.push(x);
    }

    fn len(&self) -> usize {
        self.contents.len()
    }

    fn assert_stack_size(&self, op: &str, required: usize) {
        assert!(
            self.len() >= required,
            "operation {} requires {} elements, but the stack only contains {}",
            op, required, self.len()
        );
    }
}
