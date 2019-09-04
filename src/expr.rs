
use std::collections::HashMap;

pub fn parse_func(body: &str) -> Expr {
    let tokens = tokenise(body);
    parse_expr(&tokens)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArithmeticOp { Add, Sub, Mul, Div, Exp }

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Num(f32),
    Var(String),
    Binary(ArithmeticOp, Box<Expr>, Box<Expr>),
}

impl Expr {
    pub fn evaluate(&self, vars: &HashMap<String, f32>) -> f32 {
        match self {
            Expr::Num(n) => *n,
            Expr::Var(s) => vars[s],
            Expr::Binary(op, left, right) => match op {
                ArithmeticOp::Add => left.evaluate(vars) + right.evaluate(vars),
                ArithmeticOp::Sub => left.evaluate(vars) - right.evaluate(vars),
                ArithmeticOp::Mul => left.evaluate(vars) * right.evaluate(vars),
                ArithmeticOp::Div => left.evaluate(vars) / right.evaluate(vars),
                ArithmeticOp::Exp => left.evaluate(vars).powf(right.evaluate(vars)),
            }
        }
    }
}

/// Tokens in the DSL used to define user-provided functions.
#[derive(Debug, Clone, PartialEq)]
enum Token {
    /// A named variable.
    Var(String),
    /// A named function.
    Func(String),
    Num(f32),
    Plus,
    Sub,
    Mul,
    Div,
    Pow,
    OpenParen,
    CloseParen,
}

fn tokenise(func: &str) -> Vec<Token> {
    // TODO: actually tokenise!
    //3 + 4 × 2 ÷ ( 1 − 5 ) ^ 2
    vec![
        Token::Num(3.0),
        Token::Plus,
        Token::Num(4.0),
        Token::Mul,
        Token::Num(2.0),
        Token::Div,
        Token::OpenParen,
        Token::Num(1.0),
        Token::Sub,
        Token::Num(5.0),
        Token::CloseParen,
        Token::Pow,
        Token::Num(2.0),
    ]
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum Associativity { Right, Left }

fn precedence(token: &Token) -> u32 {
    match token {
        Token::Plus => 2,
        Token::Sub => 2,
        Token::Mul => 3,
        Token::Div => 3,
        Token::Pow => 4,
        // TODO: introduce a new enum for just the tokens?
        _ => panic!("Not an operator"),
    }
}

fn associativity(token: &Token) -> Associativity {
    match token {
        Token::Plus => Associativity::Left,
        Token::Sub => Associativity::Left,
        Token::Mul => Associativity::Left,
        Token::Div => Associativity::Left,
        Token::Pow => Associativity::Right,
        // TODO: introduce a new enum for just the tokens?
        _ => panic!("Not an operator"),
    }
}

fn arithmetic_op(token: &Token) -> ArithmeticOp {
    match token {
        Token::Plus => ArithmeticOp::Add,
        Token::Sub => ArithmeticOp::Sub,
        Token::Mul => ArithmeticOp::Mul,
        Token::Div => ArithmeticOp::Div,
        Token::Pow => ArithmeticOp::Exp,
        // TODO: introduce a new enum for just the tokens?
        _ => panic!("Not an arithmetic operator"),
    }
}

fn is_function(token: &Token) -> bool {
    match token {
        Token::Func(_) => true,
        _ => false,
    }
}

fn is_lparen(token: &Token) -> bool {
    match token {
        Token::OpenParen => true,
        _ => false,
    }
}

fn is_operator(token: &Token) -> bool {
    match token {
        Token::Plus | Token::Sub | Token::Mul | Token::Div | Token::Pow => true,
        _ => false,
    }
}

fn add_node(stack: &mut Vec<Expr>, op: ArithmeticOp) {
    let right = stack.pop().unwrap();
    let left = stack.pop().unwrap();
    stack.push(Expr::Binary(op, Box::new(left), Box::new(right)));
}

/// Parses an arithmetic expression using the shunting yard algorithm.
fn parse_expr(tokens: &[Token]) -> Expr {
    let mut output_queue = Vec::new();
    let mut operator_stack = Vec::new();

    for token in tokens {
        match token {
            Token::Num(n) => output_queue.push(Expr::Num(*n)),
            Token::Var(v) => output_queue.push(Expr::Var(v.clone())),
            // TODO
            //Token::Func(_) => operator_stack.push(token.clone()),
            Token::Func(_) => panic!("Functions are not yet supported"),
            Token::Plus | Token::Sub | Token::Mul | Token::Div | Token::Pow => {
                loop {
                    if operator_stack.is_empty() {
                        break;
                    }
                    let top = &operator_stack[operator_stack.len() - 1];
                    if is_function(top)
                        || (is_operator(top) && precedence(top) > precedence(token))
                        || (is_operator(top) && precedence(top) == precedence(token) && associativity(top) == Associativity::Left)
                    {
                        let top = operator_stack.pop().unwrap();
                        add_node(&mut output_queue, arithmetic_op(&top));
                    } else {
                        break;
                    }
                }
                operator_stack.push(token.clone())
            },
            Token::OpenParen => operator_stack.push(token.clone()),
            Token::CloseParen => {
                let mut matched = false;
                loop {
                    if operator_stack.is_empty() {
                        break;
                    }
                    let top = &operator_stack[operator_stack.len() - 1];
                    if is_lparen(top) {
                        matched = true;
                        break;
                    }
                    let top = operator_stack.pop().unwrap();
                    add_node(&mut output_queue, arithmetic_op(&top));
                }
                if !matched {
                    panic!("Unmatched parenthesis");
                }
                if !operator_stack.is_empty() {
                    let top = &operator_stack[operator_stack.len() - 1];
                    if is_lparen(top) {
                        operator_stack.pop().unwrap();
                    }
                }
            }
        }
    }

    loop {
        if operator_stack.is_empty() {
            break;
        }
        let top = operator_stack.pop().unwrap();
        add_node(&mut output_queue, arithmetic_op(&top));
    }

    output_queue.pop().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Macros to make it less verbose to write test expectations involving Exprs
    macro_rules! op {
        ($op:expr, $left:expr, $right:expr) => { Expr::Binary($op, Box::new($left), Box::new($right)) }
    }
    macro_rules! add { ($left:expr, $right:expr) => {  op!(ArithmeticOp::Add, $left, $right) } }
    macro_rules! sub { ($left:expr, $right:expr) => {  op!(ArithmeticOp::Sub, $left, $right) } }
    macro_rules! mul { ($left:expr, $right:expr) => {  op!(ArithmeticOp::Mul, $left, $right) } }
    macro_rules! div { ($left:expr, $right:expr) => {  op!(ArithmeticOp::Div, $left, $right) } }
    macro_rules! exp { ($left:expr, $right:expr) => {  op!(ArithmeticOp::Exp, $left, $right) } }
    macro_rules! num { ($num:expr) => { Expr::Num($num) } }

    // Macro to make it less verbose to list Tokens in tests
    macro_rules! tokens {
        ($( $t:expr ),*) => {
            vec![$( tok($t) ),* ]
        }
    }
    // TODO: handle named functions
    fn tok(t: &str) -> Token {
        if let Ok(n) = t.parse::<f32>() {
            return Token::Num(n);
        }
        match t {
            "(" => Token::OpenParen,
            ")" => Token::CloseParen,
            "+" => Token::Plus,
            "-" => Token::Sub,
            "*" => Token::Mul,
            "/" => Token::Div,
            "^" => Token::Pow,
            _ => Token::Var(t.into())
        }
    }

    #[test]
    fn test_parse_expr_num() {
        let tokens = vec![Token::Num(2.0)];
        let expected = Expr::Num(2.0);
        let expr = parse_expr(&tokens);
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_parse_expr_var() {
        let tokens = vec![Token::Var("x".into())];
        let expected = Expr::Var("x".into());
        let expr = parse_expr(&tokens);
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_parse_expr_add() {
        // 3 + 4
        let tokens = tokens!["3", "+", "4"];
        let expected = add!(num!(3.0), num!(4.0));
        let expr = parse_expr(&tokens);
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_parse_expr_add_mul() {
        // 3 + 4 * 2
        let tokens = tokens!["3", "+", "4", "*", "2"];
        let expected = add!(num!(3.0), mul!(num!(4.0), num!(2.0)));
        let expr = parse_expr(&tokens);
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_parse_expr_add_mul_parens() {
        // (3 + 4) * 2
        let tokens = tokens!["(", "3", "+", "4", ")", "*", "2"];
        let expected = mul!(add!(num!(3.0), num!(4.0)), num!(2.0));
        let expr = parse_expr(&tokens);
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_parse_expr_mul_div() {
        // 4 x 2 / 5
        let tokens = tokens!["4", "*", "2", "/", "5"];
        let expected = div!(mul!(num!(4.0), num!(2.0)), num!(5.0));
        let expr = parse_expr(&tokens);
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_parse_expr_complex() {
        // 3 + 4 × 2 ÷ ( 1 − 5 ) ^ 2
        let tokens = tokens!["3", "+", "4", "*", "2", "/", "(", "1", "-", "5", ")", "^", "2"];
        let expected = add!(
            num!(3.0),
            div!(
                mul!(num!(4.0), num!(2.0)),
                exp!(
                    sub!(num!(1.0), num!(5.0)),
                    num!(2.0)
                )
            )
        );
        let expr = parse_expr(&tokens);
        assert_eq!(expr, expected);
    }
}
