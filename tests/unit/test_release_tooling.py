"""Release tooling: SBOM generation and the clean-install gate (unit)."""


from scripts_helper import load_script  # noqa: F401  (sys.path bootstrap)

generate_sbom = load_script("generate_sbom.py")


def test_sbom_structure_and_closure():
    sbom, hint = generate_sbom.build_sbom()
    assert sbom["bomFormat"] == "CycloneDX"
    assert sbom["specVersion"] == "1.5"
    assert sbom["serialNumber"].startswith("urn:uuid:")
    names = [c["name"] for c in sbom["components"]]
    assert len(names) == len(set(names)), "duplicate components in SBOM"
    for root in ("torch", "numpy", "safetensors", "tokenizers"):
        assert root in names, f"root dependency {root} missing from SBOM"
    apps = [c for c in sbom["components"] if c["name"] == "laya-linux"]
    assert len(apps) == 1 and apps[0]["type"] == "application"
    assert apps[0]["version"] == "0.1.0"
    refs = [c["bom-ref"] for c in sbom["components"]]
    assert len(refs) == len(set(refs))
    assert all(c.get("purl", "").startswith("pkg:pypi/") for c in sbom["components"])
    assert hint is None  # the dev environment has the editable install
