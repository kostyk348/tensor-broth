from tensor_broth.benchmark import compare


def test_compare_smoke():
    report = compare(cube=12, steps=5, vm_steps=2_000)
    base = report.vm_baseline
    assert base.vm_mops > 0
    assert len(report.results) == 1
    r = report.results[0]
    assert r.soup_gops > 0
    assert 0 <= r.soup_activity <= 1


def test_markdown_table_renders():
    report = compare(cube=10, steps=3, vm_steps=1_000)
    table = report.markdown()
    assert table.splitlines()[0].startswith("| backend")
    assert "vm" in table and ("torch" in table or "numpy" in table)
