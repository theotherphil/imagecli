
use nom::{
    IResult,
    bytes::complete::tag,
    combinator::{map, map_res, opt},
    character::complete::{digit1, space1},
    sequence::{preceded, tuple},
};

/// Parses a integer value.
pub fn int<T: std::str::FromStr>(input: &str) -> IResult<&str, T> {
    map_res(digit1, |s: &str| s.parse::<T>())(input)
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
            preceded(space1, arg1)
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
