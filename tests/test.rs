use imagecli::image_ops::parse_pipeline;

fn assert_pipeline_parse(pipeline: &str, expected: &[&str]) {
    let parsed = parse_pipeline(pipeline);
    match parsed {
        Err(_) => panic!("Parse should succeed"),
        Ok((i, v)) => {
            assert_eq!(i, "");
            let descriptions: Vec<String> = v.iter().map(|o| format!("{:?}", o)).collect();
            assert_eq!(descriptions, expected);
        }
    }
}

fn assert_pipeline_parse_failure(pipeline: &str) {
    let parsed = parse_pipeline(pipeline);
    match parsed {
        Ok((i, v)) => panic!(
            "Parse succeeded, but should have failed. i: {:?}, v: {:?}",
            i, v
        ),
        Err(_) => (), // TODO: add error reporting, with tests
    }
}

#[test]
fn test_parse_valid_single_stage_pipeline() {
    assert_pipeline_parse(
            "circle filled 231 337 100 (255, 255, 0)",
            &vec!["Circle { fill: Filled, center: (231, 337), radius: 100, color: Rgb(Rgb([255, 255, 0])) }"]
        );
}

#[test]
fn test_parse_invalid_single_stage_pipeline() {
    // Not valid, as grid requires two numerical inputs)
    assert_pipeline_parse_failure("grid 1");
}

#[test]
fn test_parse_valid_multi_stage_pipeline() {
    assert_pipeline_parse(
        "grid 1 2 > gray > scale 3 > id",
        &vec!["Grid(1, 2)", "Gray", "Scale(3.0)", "Id"],
    );
}

#[test]
fn test_parse_invalid_multi_stage_pipeline() {
    // The grid stage is invalid
    assert_pipeline_parse_failure("gray > grid 1 > scale 3 > id");
}

#[test]
fn test_parse_circle() {
    assert_pipeline_parse(
            "circle filled 231 337 100 (255, 255, 0)",
            &vec!["Circle { fill: Filled, center: (231, 337), radius: 100, color: Rgb(Rgb([255, 255, 0])) }"]
        );
}

// TODO: the error messages from this parser are going to be pretty useless.
#[test]
fn test_parse_resize() {
    // Need to provide at least one of width or height
    assert_pipeline_parse_failure("resize");
    // If providing only one parameter then need to specify
    // whether it's width or height
    assert_pipeline_parse_failure("resize 10");
    // Providing both positional arguments
    assert_pipeline_parse(
        "resize 10 12",
        &vec!["Resize { width: Some(10), height: Some(12) }"],
    );
    // Providing both named arguments
    assert_pipeline_parse(
        "resize w=10 h=12",
        &vec!["Resize { width: Some(10), height: Some(12) }"],
    );
    // Providing both named arguments in opposite order
    assert_pipeline_parse(
        "resize h=12 w=10",
        &vec!["Resize { width: Some(10), height: Some(12) }"],
    );
    // Width only
    assert_pipeline_parse(
        "resize w=10",
        &vec!["Resize { width: Some(10), height: None }"],
    );
    // Height only
    assert_pipeline_parse(
        "resize h=12",
        &vec!["Resize { width: None, height: Some(12) }"],
    );
}

#[test]
fn test_parse_hcat() {
    // No width provided - default to 2
    assert_pipeline_parse("hcat", &vec!["Grid(2, 1)"]);
    // Width provided
    assert_pipeline_parse("hcat 3", &vec!["Grid(3, 1)"]);
}

#[test]
fn test_parse_scale() {
    // No number provided
    assert_pipeline_parse_failure("scale");
    // Correct incantation, integer literal
    assert_pipeline_parse("scale 3", &vec!["Scale(3.0)"]);
    // Correct incantation, floating point literal
    assert_pipeline_parse("scale 3.0", &vec!["Scale(3.0)"]);
}

#[test]
fn test_parse_gray() {
    assert_pipeline_parse("gray", &vec!["Gray"]);
}

#[test]
fn test_parse_grid() {
    // Only one number provided
    assert_pipeline_parse_failure("grid 12");
    // Correct incantation
    assert_pipeline_parse("grid 12 34", &vec!["Grid(12, 34)"]);
}

#[test]
fn test_parse_translate_negative() {
    assert_pipeline_parse("translate -10 -20", &vec!["Translate(-10, -20)"]);
}
