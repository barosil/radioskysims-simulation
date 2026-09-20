def test_resource_aware_processor_is_public() -> None:
    from radioskysims_simulation import ResourceAwareProcessor

    assert ResourceAwareProcessor is not None
