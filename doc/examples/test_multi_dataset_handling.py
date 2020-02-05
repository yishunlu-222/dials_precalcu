from multi_dataset_handling import split_reflection_tables, handling_missing_data


def test_handle_data(tmpdir):
    """Test the code in the documentation"""
    with tmpdir.as_cwd():
        split_reflection_tables()


def test_handle_missing_data(tmpdir):
    """Test the code in the documentation"""
    with tmpdir.as_cwd():
        handling_missing_data()
