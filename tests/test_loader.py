from rag.loader import load_documents


def test_loads_supported_files_only(tmp_path):
    (tmp_path / "a.txt").write_text("alpha text", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.md").write_text("# beta", encoding="utf-8")
    (tmp_path / "c.csv").write_text("ignored,file", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("   ", encoding="utf-8")

    docs = load_documents(tmp_path)

    assert [d["source"] for d in docs] == ["a.txt", "sub/b.md"]
    assert docs[0]["text"] == "alpha text"
