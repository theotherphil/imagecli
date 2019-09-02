
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

    /// duplicates the top element of the stack n times.
    ///
    /// dup 1 (a -- a a)
    /// dup 2 (a -- a a a)
    pub fn dup(&mut self, n: usize) {
        self.assert_stack_size("dup", 1);
        // TODO: Remove unnecessary clone
        let a = self.contents[self.len() - 1].clone();
        for _ in 0..n {
            self.contents.push(a.clone());
        }
    }

    /// drop ( a -- )
    pub fn drop(&mut self) {
        self.assert_stack_size("drop", 1);
        self.contents.remove(self.len() - 1);
    }

    /// rotates the top n elements of the stack.
    /// rot(1) is a no-op, rot(2) swaps the top two elements.
    pub fn rot(&mut self, n: usize) {
        if n < 2 { return; }
        self.assert_stack_size("rot", n);
        // TODO: Remove unnecessary clone
        let a = self.contents[self.len() - 1].clone();
        self.contents.remove(self.contents.len() - 1);
        self.contents.insert(self.len() - (n - 1), a);
    }

    /// over ( a b -- a b a )
    pub fn over(&mut self) {
        self.assert_stack_size("over", 2);
        let a = self.contents[self.len() - 1].clone();
        self.contents.insert(self.len() - 2, a);
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
