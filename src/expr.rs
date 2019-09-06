
use nom::{
    IResult,
    branch::alt,
    bytes::complete::tag,
    combinator::map,
    character::complete::{alpha1, space0},
    multi::many0,
    number::complete::float,
    sequence::{delimited, pair, preceded},
};

pub fn parse_func<'a, 'b>(input: &'a str, name: &'b str) -> IResult<&'a str, (String, Expr)> {
    let (i, tokens) = lex(input, name)?;
    let body: Vec<String> = tokens.iter().map(|t| t.to_string()).collect();
    Ok((i, (body.join(" "), parse_expr(&tokens))))
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
    pub fn evaluate(&self, x: f32, y: f32, p: f32, q: f32) -> f32 {
        match self {
            Expr::Num(n) => *n,
            Expr::Var(s) => match s.as_ref() {
                "x" => x,
                "y" => y,
                "p" => p,
                "q" => q,
                _ => panic!("Invalid variable"),
            },
            Expr::Binary(op, left, right) => match op {
                ArithmeticOp::Add => left.evaluate(x, y, p, q) + right.evaluate(x, y, p, q),
                ArithmeticOp::Sub => left.evaluate(x, y, p, q) - right.evaluate(x, y, p, q),
                ArithmeticOp::Mul => left.evaluate(x, y, p, q) * right.evaluate(x, y, p, q),
                ArithmeticOp::Div => left.evaluate(x, y, p, q) / right.evaluate(x, y, p, q),
                ArithmeticOp::Exp => left.evaluate(x, y, p, q).powf(right.evaluate(x, y, p, q)),
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

impl std::fmt::Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Token::Var(v) => v.clone(),
            Token::Func(b) => b.clone(),
            Token::Num(n) => n.to_string(),
            Token::Plus => "+".into(),
            Token::Sub => "-".into(),
            Token::Mul => "*".into(),
            Token::Div => "/".into(),
            Token::Pow => "^".into(),
            Token::OpenParen => "(".into(),
            Token::CloseParen => ")".into(),
        };
        write!(f, "{}", s)
    }
}

fn token(input: &str) -> IResult<&str, Token> {
    // TODO: handle named functions
    alt((
        map(tag("+"), |_| Token::Plus),
        map(tag("-"), |_| Token::Sub),
        map(tag("*"), |_| Token::Mul),
        map(tag("/"), |_| Token::Div),
        map(tag("^"), |_| Token::Pow),
        map(tag("("), |_| Token::OpenParen),
        map(tag(")"), |_| Token::CloseParen),
        map(alpha1, |n: &str| Token::Var(n.into())),
        map(float, |f| Token::Num(f)),
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
                space0),
            tag("}"),
        )
    )(input)
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
    macro_rules! var { ($var:expr) => { Expr::Var($var.into()) } }

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
    fn test_lex() {
        assert_eq!(
            lex("func { p + x / 5 + y / 5 }", "func"),
            Ok(("", vec![
                Token::Var("p".into()),
                Token::Plus,
                Token::Var("x".into()),
                Token::Div,
                Token::Num(5.0),
                Token::Plus,
                Token::Var("y".into()),
                Token::Div,
                Token::Num(5.0),
            ]))
        );
        assert_eq!(
            lex("func2 { (p + q) / 2 }", "func2"),
            Ok(("", vec![
                Token::OpenParen,
                Token::Var("p".into()),
                Token::Plus,
                Token::Var("q".into()),
                Token::CloseParen,
                Token::Div,
                Token::Num(2.0),
            ])),
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
        let expected = add!(add!(var!("p"), div!(var!("x"), num!(10.0))), div!(var!("y"), num!(10.0)));
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
