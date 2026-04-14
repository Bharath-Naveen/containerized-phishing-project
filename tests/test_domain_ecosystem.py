from src.app_v1.domain_ecosystem import domain_relation


def test_same_registrable_domain() -> None:
    rel = domain_relation("medium.com", "help.medium.com")
    assert rel["same_registrable_domain"] is True
    assert rel["truly_external"] is False


def test_cross_registrable_wiki_ecosystem() -> None:
    rel = domain_relation("wikipedia.org", "wikimedia.org")
    assert rel["same_trusted_ecosystem"] is True
    assert rel["truly_external"] is False


def test_userinfo_like_evil_domain_is_external() -> None:
    rel = domain_relation("reddit.com", "evil.xyz")
    assert rel["truly_external"] is True
