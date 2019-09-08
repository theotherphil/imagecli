
use nom::{
    IResult,
    bytes::complete::tag,
    combinator::{map, map_res, opt},
    character::complete::{digit1, space0, space1},
    multi::separated_nonempty_list,
    sequence::{delimited, pair, preceded, tuple},
};

/// Converts a parser that returns a T: SomeTrait to one that returns a Box<dyn SomeTrait>.
#[macro_export]
macro_rules! map_to_boxed_trait {
    ($parser:expr, $trait:ident) => {
        map($parser, |o| {
            let boxed: Box<dyn $trait> = Box::new(o);
            boxed
        })
    }
}

/// Parses a nonempty separated list where the separator can be padded with arbitrary whitespace.
pub fn nonempty_sequence<'a, T, F>(
    separator: &'static str,
    element: F
) -> impl Fn(&'a str) -> IResult<&'a str, Vec<T>>
where
    F: Fn(&'a str) -> IResult<&'a str, T>
{
    separated_nonempty_list(
        delimited(space0, tag(separator), space0),
        element
    )
}

/// Parses a integer value.
pub fn int<T: std::str::FromStr>(input: &str) -> IResult<&str, T> {
    map_res(digit1, |s: &str| s.parse::<T>())(input)
}

/// Matches 'name=<arg>'.
pub fn named_arg<'a, F, T>(
    name: &'static str,
    arg: F
) -> impl Fn(&'a str) -> IResult<&'a str, T>
where
    F: Fn(&'a str) -> IResult<&'a str, T>
{
    preceded(
        pair(tag(name), tag("=")),
        arg
    )
}

/// Parser for an operator which takes no args.
pub fn op_zero<T: Clone>(name: &'static str, t: T) -> impl Fn(&str) -> IResult<&str, T> {
    move |i| map(tag(name), |_| t.clone())(i)
}

/// Parser for an operator which takes a single arg.
pub fn op_one<'a, T, F1, A1, G>(
    name: &'static str,
    arg1: F1,
    build: G
) -> impl Fn(&'a str) -> IResult<&'a str, T>
where
    F1: Fn(&'a str) -> IResult<&'a str, A1> + Copy,
    G: Fn(A1) -> T + Copy
{
    move |i| map(
        preceded(
            tag(name),
            preceded(space1, arg1),
        ),
        |val| build(val)
    )(i)
}

/// Parser for an operator which takes a single optional arg.
pub fn op_one_opt<'a, T, F1, A1, G>(
    name: &'static str,
    arg1: F1,
    build: G
) -> impl Fn(&'a str) -> IResult<&'a str, T>
where
    F1: Fn(&'a str) -> IResult<&'a str, A1> + Copy,
    G: Fn(Option<A1>) -> T + Copy
{
    move |i| map(
        preceded(
            tag(name),
            opt(preceded(space1, arg1))
        ),
        build
    )(i)
}

/// Parser for an operator which takes two args.
pub fn op_two<T, F1, F2, A1, A2, G>(
    name: &'static str,
    arg1: F1,
    arg2: F2,
    build: G
) -> impl Fn(&str) -> IResult<&str, T>
where
    F1: Fn(&str) -> IResult<&str, A1> + Copy,
    F2: Fn(&str) -> IResult<&str, A2> + Copy,
    G: Fn(A1, A2) -> T + Copy
{
    move |i| map(
        preceded(
            tag(name),
            tuple((
                preceded(space1, arg1),
                preceded(space1, arg2),
            ))
        ),
        |(val1, val2)| build(val1, val2)
    )(i)
}
