from backend.security import hash_password, verify_password


def test_password_hash_is_salted_and_verifiable() -> None:
    first = hash_password("a-long-password-for-tests")
    second = hash_password("a-long-password-for-tests")

    assert first != second
    assert verify_password("a-long-password-for-tests", first)
    assert not verify_password("incorrect-password", first)


def test_password_verification_rejects_malformed_hash() -> None:
    assert not verify_password("anything", "not-a-valid-hash")
