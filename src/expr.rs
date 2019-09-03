
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

#[test]
fn test_expr() {
    // 3 + 4 × 2 ÷ ( 1 − 5 ) ^ 2
    // TODO!
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
    let tokens = vec![
        Token::Num(3.0),
        Token::Plus,
        Token::Num(4.0),
    ];
    let expected = Expr::Binary(
        ArithmeticOp::Add,
        Box::new(Expr::Num(3.0)),
        Box::new(Expr::Num(4.0)),
    );
    let expr = parse_expr(&tokens);
    assert_eq!(expr, expected);
}

#[test]
fn test_parse_expr_add_mul() {
    // 3 + 4 * 2
    let tokens = vec![
        Token::Num(3.0),
        Token::Plus,
        Token::Num(4.0),
        Token::Mul,
        Token::Num(2.0),
    ];
    let expected = Expr::Binary(
        ArithmeticOp::Add,
        Box::new(Expr::Num(3.0)),
        Box::new(
            Expr::Binary(
                ArithmeticOp::Mul,
                Box::new(Expr::Num(4.0)),
                Box::new(Expr::Num(2.0))
            )
        )
    );
    let expr = parse_expr(&tokens);
    assert_eq!(expr, expected);
}

#[test]
fn test_parse_expr_add_mul_parens() {
    // (3 + 4) * 2
    let tokens = vec![
        Token::OpenParen,
        Token::Num(3.0),
        Token::Plus,
        Token::Num(4.0),
        Token::CloseParen,
        Token::Mul,
        Token::Num(2.0),
    ];
    let expected = Expr::Binary(
        ArithmeticOp::Mul,
        Box::new(
            Expr::Binary(
                ArithmeticOp::Add,
                Box::new(Expr::Num(3.0)),
                Box::new(Expr::Num(4.0))
            )
        ),
        Box::new(Expr::Num(2.0)),
    );
    let expr = parse_expr(&tokens);
    assert_eq!(expr, expected);
}

#[test]
fn test_parse_expr_complex() {
    //3 + 4 × 2 ÷ ( 1 − 5 ) ^ 2
    let tokens = vec![
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
    ];
    
    //3 + 4 × 2 ÷ ( 1 − 5 ) ^ 2
    let expected = Expr::Binary(
        ArithmeticOp::Add,
        Box::new(Expr::Num(3.0)),
        Box::new(
            Expr::Binary(
                ArithmeticOp::Div,
                Box::new(
                    Expr::Binary(
                        ArithmeticOp::Mul,
                        Box::new(Expr::Num(4.0)),
                        Box::new(Expr::Num(2.0))
                    )
                ),
                Box::new(
                    Expr::Binary(
                        ArithmeticOp::Exp,
                        Box::new(
                            Expr::Binary(
                                ArithmeticOp::Sub,
                                Box::new(Expr::Num(1.0)),
                                Box::new(Expr::Num(5.0))
                            )
                        ),
                        Box::new(Expr::Num(2.0))
                    )
                )
            )
        )
    );
    let expr = parse_expr(&tokens);
    assert_eq!(expr, expected);
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum Associativity { Right, Left }

fn precedence(token: &Token) -> u32 {
    match token {
        Token::Plus => 2,
        Token::Sub => 2,
        Token::Mul => 3,
        Token::Div => 4,
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
    //     read a token.
        match token {
    //     if the token is a number, then:
    //         push it to the output queue.
            Token::Num(n) => output_queue.push(Expr::Num(*n)),
            Token::Var(v) => output_queue.push(Expr::Var(v.clone())),
    //     if the token is a function then:
    //         push it onto the operator stack 

            // TODO
            //Token::Func(_) => operator_stack.push(token.clone()),
            Token::Func(_) => panic!("Functions are not yet supported"),

    //     if the token is an operator, then:
    //         while (
    //          (
    //              (there is a function at the top of the operator stack)
    //              or (there is an operator at the top of the operator stack with greater precedence)
    //              or (the operator at the top of the operator stack has equal precedence and is left associative)
    //          )
    //          and (the operator at the top of the operator stack is not a left parenthesis
    //         ):
    //             pop operators from the operator stack onto the output queue.
    //         push it onto the operator stack.
            Token::Plus | Token::Sub | Token::Mul | Token::Div | Token::Pow => {
                loop {
                    if operator_stack.is_empty() {
                        break;
                    }
                    let top = &operator_stack[operator_stack.len() - 1];
                    if (
                            is_function(top)
                            || (is_operator(top) && precedence(top) > precedence(token))
                            || (is_operator(top) && precedence(top) == precedence(token) && associativity(top) == Associativity::Left)
                        ) // && !is_lparen(top) <-- in wiki article, but redundant. Suspicious...
                    {
                        let top = operator_stack.pop().unwrap();
                        add_node(&mut output_queue, arithmetic_op(&top));
                    } else {
                        break;
                    }
                }
                operator_stack.push(token.clone())
            },
    //     if the token is a left paren (i.e. "("), then:
    //         push it onto the operator stack.
            Token::OpenParen => operator_stack.push(token.clone()),
    //     if the token is a right paren (i.e. ")"), then:
    //         while the operator at the top of the operator stack is not a left paren:
    //             pop the operator from the operator stack onto the output queue.
    //         /* if the stack runs out without finding a left paren, then there are mismatched parentheses. */
    //         if there is a left paren at the top of the operator stack, then:
    //             pop the operator from the operator stack and discard it
            Token::CloseParen => {
                loop {
                    if operator_stack.is_empty() {
                        break;
                    }
                    let top = &operator_stack[operator_stack.len() - 1];
                    if is_lparen(top) {
                        break;
                    }
                    let top = operator_stack.pop().unwrap();
                    add_node(&mut output_queue, arithmetic_op(&top));
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

    // after while loop, if operator stack not null, pop everything to output queue
    loop {
        if operator_stack.is_empty() {
            break;
        }
        let top = operator_stack.pop().unwrap();
        add_node(&mut output_queue, arithmetic_op(&top));
    }

    // if there are no more tokens to read then:
    //     while there are still operator tokens on the stack:
    //         /* if the operator token on the top of the stack is a paren, then there are mismatched parentheses. */
    //         pop the operator from the operator stack onto the output queue.
    
    // TODO: ^^ ??? I've already read all the tokens and popped everything from the stack... 
    
    println!("OUTPUT QUEUE: {:#?}", output_queue);
    output_queue.pop().unwrap()
}
