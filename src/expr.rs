use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{alpha1, space0},
    combinator::map,
    multi::many0,
    number::complete::float,
    sequence::{delimited, pair, preceded},
    IResult,
};

pub fn parse_func<'a, 'b>(input: &'a str, name: &'b str) -> IResult<&'a str, (String, Expr)> {
    let (i, tokens) = lex(input, name)?;
    let body: Vec<String> = tokens.iter().map(|t| t.to_string()).collect();
    Ok((i, (body.join(" "), parse_expr(&tokens))))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArithmeticOp {
    Add,
    Sub,
    Mul,
    Div,
    Exp,
    Gt,
    Lt,
    Eq,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Num(f32),
    Var(String),
    Binary(ArithmeticOp, Box<Expr>, Box<Expr>),
}

impl Expr {
    pub fn evaluate(&self, x: f32, y: f32, p: f32, q: f32, r: f32) -> f32 {
        match self {
            Expr::Num(n) => *n,
            Expr::Var(s) => match s.as_ref() {
                "x" => x,
                "y" => y,
                "p" => p,
                "q" => q,
                "r" => r,
                _ => panic!("Invalid variable"),
            },
            Expr::Binary(op, left, right) => {
                let (left, right) = (left.evaluate(x, y, p, q, r), right.evaluate(x, y, p, q, r));
                match op {
                    ArithmeticOp::Add => left + right,
                    ArithmeticOp::Sub => left - right,
                    ArithmeticOp::Mul => left * right,
                    ArithmeticOp::Div => left / right,
                    ArithmeticOp::Exp => left.powf(right),
                    ArithmeticOp::Gt => {
                        if left > right {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    ArithmeticOp::Lt => {
                        if left < right {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    ArithmeticOp::Eq => {
                        if left == right {
                            1.0
                        } else {
                            0.0
                        }
                    }
                }
            }
        }
    }
}

/// Tokens in the DSL used to define user-provided functions.
#[derive(Debug, Clone, PartialEq)]
enum Token {
    /// A named variable.
    Var(String),
    Num(f32),
    Add,
    Sub,
    Mul,
    Div,
    Exp,
    Gt,
    Lt,
    Eq,
    LParen,
    RParen,
}

impl std::fmt::Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Token::Var(v) => v.clone(),
            Token::Num(n) => n.to_string(),
            Token::Add => "+".into(),
            Token::Sub => "-".into(),
            Token::Mul => "*".into(),
            Token::Div => "/".into(),
            Token::Exp => "^".into(),
            Token::Gt => ">".into(),
            Token::Lt => "<".into(),
            Token::Eq => "=".into(),
            Token::LParen => "(".into(),
            Token::RParen => ")".into(),
        };
        write!(f, "{}", s)
    }
}

fn token(input: &str) -> IResult<&str, Token> {
    alt((
        map(tag("+"), |_| Token::Add),
        map(tag("-"), |_| Token::Sub),
        map(tag("*"), |_| Token::Mul),
        map(tag("/"), |_| Token::Div),
        map(tag("^"), |_| Token::Exp),
        map(tag(">"), |_| Token::Gt),
        map(tag("<"), |_| Token::Lt),
        map(tag("="), |_| Token::Eq),
        map(tag("("), |_| Token::LParen),
        map(tag(")"), |_| Token::RParen),
        map(alpha1, |n: &str| Token::Var(n.into())),
        map(float, Token::Num),
    ))(input)
}

// Creates tokens from a func definition.
// 'func { .. }'
fn lex<'a, 'b>(input: &'a str, name: &'b str) -> IResult<&'a str, Vec<Token>> {
    preceded(
        pair(tag(name), space0),
        delimited(
            tag("{"),
            delimited(
                space0,
                // separated_list(space0, token) does not work here
                many0(preceded(space0, token)),
                space0,
            ),
            tag("}"),
        ),
    )(input)
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum Associativity {
    Right,
    Left,
}

// Higher precedence operators bind more tightly
fn precedence(token: &Token) -> u32 {
    match token {
        Token::Gt => 1,
        Token::Lt => 1,
        Token::Eq => 1,
        Token::Add => 2,
        Token::Sub => 2,
        Token::Mul => 3,
        Token::Div => 3,
        Token::Exp => 4,
        // TODO: introduce a new enum for just the tokens?
        _ => panic!("Not an operator"),
    }
}

fn associativity(token: &Token) -> Associativity {
    match token {
        Token::Gt => Associativity::Left,
        Token::Lt => Associativity::Left,
        Token::Eq => Associativity::Left,
        Token::Add => Associativity::Left,
        Token::Sub => Associativity::Left,
        Token::Mul => Associativity::Left,
        Token::Div => Associativity::Left,
        Token::Exp => Associativity::Right,
        // TODO: introduce a new enum for just the ops?
        _ => panic!("Not an operator"),
    }
}

fn arithmetic_op(token: &Token) -> ArithmeticOp {
    match token {
        Token::Gt => ArithmeticOp::Gt,
        Token::Lt => ArithmeticOp::Lt,
        Token::Eq => ArithmeticOp::Eq,
        Token::Add => ArithmeticOp::Add,
        Token::Sub => ArithmeticOp::Sub,
        Token::Mul => ArithmeticOp::Mul,
        Token::Div => ArithmeticOp::Div,
        Token::Exp => ArithmeticOp::Exp,
        // TODO: introduce a new enum for just the ops?
        _ => panic!("Not an arithmetic operator"),
    }
}

fn is_lparen(token: &Token) -> bool {
    match token {
        Token::LParen => true,
        _ => false,
    }
}

fn is_operator(token: &Token) -> bool {
    match token {
        Token::Add
        | Token::Sub
        | Token::Mul
        | Token::Div
        | Token::Exp
        | Token::Lt
        | Token::Gt
        | Token::Eq => true,
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
            Token::Add
            | Token::Sub
            | Token::Mul
            | Token::Div
            | Token::Exp
            | Token::Lt
            | Token::Gt
            | Token::Eq => {
                loop {
                    if operator_stack.is_empty() {
                        break;
                    }
                    let top = &operator_stack[operator_stack.len() - 1];
                    if (is_operator(top) && precedence(top) > precedence(token))
                        || (is_operator(top)
                            && precedence(top) == precedence(token)
                            && associativity(top) == Associativity::Left)
                    {
                        let top = operator_stack.pop().unwrap();
                        add_node(&mut output_queue, arithmetic_op(&top));
                    } else {
                        break;
                    }
                }
                operator_stack.push(token.clone())
            }
            Token::LParen => operator_stack.push(token.clone()),
            Token::RParen => {
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
        ($op:expr, $left:expr, $right:expr) => {
            Expr::Binary($op, Box::new($left), Box::new($right))
        };
    }
    macro_rules! add {
        ($left:expr, $right:expr) => {
            op!(ArithmeticOp::Add, $left, $right)
        };
    }
    macro_rules! sub {
        ($left:expr, $right:expr) => {
            op!(ArithmeticOp::Sub, $left, $right)
        };
    }
    macro_rules! mul {
        ($left:expr, $right:expr) => {
            op!(ArithmeticOp::Mul, $left, $right)
        };
    }
    macro_rules! div {
        ($left:expr, $right:expr) => {
            op!(ArithmeticOp::Div, $left, $right)
        };
    }
    macro_rules! exp {
        ($left:expr, $right:expr) => {
            op!(ArithmeticOp::Exp, $left, $right)
        };
    }
    macro_rules! gt {
        ($left:expr, $right:expr) => {
            op!(ArithmeticOp::Gt, $left, $right)
        };
    }
    macro_rules! num {
        ($num:expr) => {
            Expr::Num($num)
        };
    }
    macro_rules! var {
        ($var:expr) => {
            Expr::Var($var.into())
        };
    }

    // Macro to make it less verbose to list Tokens in tests
    macro_rules! tokens {
        ($( $t:expr ),*) => {
            vec![$( tok($t) ),* ]
        }
    }
    fn tok(t: &str) -> Token {
        if let Ok(n) = t.parse::<f32>() {
            return Token::Num(n);
        }
        match t {
            "(" => Token::LParen,
            ")" => Token::RParen,
            "+" => Token::Add,
            "-" => Token::Sub,
            "*" => Token::Mul,
            "/" => Token::Div,
            "^" => Token::Exp,
            ">" => Token::Gt,
            "<" => Token::Lt,
            "=" => Token::Eq,
            _ => Token::Var(t.into()),
        }
    }

    #[test]
    fn test_lex() {
        assert_eq!(
            lex("func { p + x / 5 + y / 5 }", "func"),
            Ok((
                "",
                vec![
                    Token::Var("p".into()),
                    Token::Add,
                    Token::Var("x".into()),
                    Token::Div,
                    Token::Num(5.0),
                    Token::Add,
                    Token::Var("y".into()),
                    Token::Div,
                    Token::Num(5.0),
                ]
            ))
        );
        assert_eq!(
            lex("func2 { (p + q) / 2 }", "func2"),
            Ok((
                "",
                vec![
                    Token::LParen,
                    Token::Var("p".into()),
                    Token::Add,
                    Token::Var("q".into()),
                    Token::RParen,
                    Token::Div,
                    Token::Num(2.0),
                ]
            )),
        );
    }

    #[test]
    fn test_parse_expr_num() {
        let tokens = tokens!["2"];
        let expected = num!(2.0);
        let expr = parse_expr(&tokens);
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_parse_expr_var() {
        let tokens = vec![Token::Var("x".into())];
        let expected = var!("x");
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
    fn test_parse_expr_multiple_vars() {
        // p + x / 10 + y / 10
        let tokens = tokens!["p", "+", "x", "/", "10", "+", "y", "/", "10"];
        let expected = add!(
            add!(var!("p"), div!(var!("x"), num!(10.0))),
            div!(var!("y"), num!(10.0))
        );
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
                exp!(sub!(num!(1.0), num!(5.0)), num!(2.0))
            )
        );
        let expr = parse_expr(&tokens);
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_parse_expr_relational() {
        // (3 + 4) * 2 > 1
        let tokens = tokens!["(", "3", "+", "4", ")", "*", "2", ">", "1"];
        let expected = gt!(mul!(add!(num!(3.0), num!(4.0)), num!(2.0)), num!(1.0));
        let expr = parse_expr(&tokens);
        assert_eq!(expr, expected);
    }
}
