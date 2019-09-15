
/// A stack supporting basic manipulations required for
/// stack-based programming.
pub struct Stack<T> {
    // The top of the stack is the last element in the vector.
    contents: Vec<T>,
}

impl<T: Clone> Stack<T> {
    /// Create a stack with the given initial elements.
    /// The top of the stack is the first element of the input.
    pub fn new(contents: Vec<T>) -> Self {
        let contents = contents.into_iter().rev().collect();
        Stack { contents }
    }

    /// Consume the stack and return its elements.
    /// The top of the stack is the first element of the output.
    pub fn contents(self) -> Vec<T> {
        self.contents.into_iter().rev().collect()
    }

    /// duplicates the top element of the stack n times.
    ///
    /// dup 1 (a -- a a)
    /// dup 2 (a -- a a a)
    pub fn dup(&mut self, n: usize) {
        self.assert_stack_size("dup", 1);
        let a = self.contents.pop().unwrap();
        for _ in 0..n {
            self.contents.push(a.clone());
        }
        self.contents.push(a);
    }

    /// rotates the top n elements of the stack.
    /// rot(1) is a no-op, rot(2) swaps the top two elements.
    pub fn rot(&mut self, n: usize) {
        if n < 2 {
            return;
        }
        self.assert_stack_size("rot", n);
        let a = self.contents.remove(self.contents.len() - 1);
        self.contents.insert(self.len() - (n - 1), a);
    }

    /// pops the top of the stack.
    pub fn pop(&mut self) -> T {
        self.assert_stack_size("pop", 1);
        self.contents.pop().unwrap()
    }

    /// pops the to n elements of the stack.
    pub fn pop_n(&mut self, n: usize) -> Vec<T> {
        self.assert_stack_size("pop_n", n);
        // TODO: remove unnecessary work
        let mut popped = self.contents.split_off(self.len() - n);
        popped.reverse();
        popped
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
            op,
            required,
            self.len()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_contents() {
        let stack = Stack::new(vec![1, 2, 3]);
        let contents = stack.contents();
        assert_eq!(contents, vec![1, 2, 3]);
    }

    #[test]
    fn test_pop() {
        let mut stack = Stack::new(vec![1, 2]);
        assert_eq!(stack.len(), 2);
        assert_eq!(stack.pop(), 1);
        assert_eq!(stack.len(), 1);
        assert_eq!(stack.pop(), 2);
    }

    #[test]
    #[should_panic]
    fn test_pop_empty() {
        let mut stack = Stack::new(vec![1]);
        assert_eq!(stack.len(), 1);
        assert_eq!(stack.pop(), 1);
        let _ = stack.pop();
    }

    #[test]
    fn test_dup() {
        let mut stack = Stack::new(vec![10, 11]);
        stack.dup(1);
        assert_eq!(stack.pop(), 10);
        assert_eq!(stack.pop(), 10);
        assert_eq!(stack.pop(), 11);
        stack.push(12);
        stack.dup(2);
        assert_eq!(stack.pop(), 12);
        assert_eq!(stack.pop(), 12);
        assert_eq!(stack.pop(), 12);
        assert_eq!(stack.len(), 0);
    }

    #[test]
    fn test_rot() {
        let mut stack = Stack::new(vec![1, 2, 3]);
        stack.rot(0);
        assert_eq!(stack.contents(), vec![1, 2, 3]);

        let mut stack = Stack::new(vec![1, 2, 3]);
        stack.rot(1);
        assert_eq!(stack.contents(), vec![1, 2, 3]);

        let mut stack = Stack::new(vec![1, 2, 3]);
        stack.rot(2);
        assert_eq!(stack.contents(), vec![2, 1, 3]);

        let mut stack = Stack::new(vec![1, 2, 3]);
        stack.rot(3);
        assert_eq!(stack.contents(), vec![2, 3, 1]);
    }

    #[test]
    #[should_panic]
    fn test_rot_exceeding_len() {
        let mut stack = Stack::new(vec![1, 2, 3]);
        stack.rot(4);
    }

    #[test]
    fn test_pop_n() {
        let mut stack = Stack::new(vec![1, 2, 3]);
        let top = stack.pop_n(2);
        assert_eq!(top, vec![1, 2]);
        assert_eq!(stack.contents(), vec![3]);
    }
}
