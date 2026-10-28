"""Set OSHC_TEST_NO_NETWORK=1 to prove tests do not fetch data or weights."""
import os
import socket

import pytest


@pytest.fixture(autouse=True)
def no_network_in_offline_test_mode(monkeypatch):
    if os.environ.get("OSHC_TEST_NO_NETWORK") != "1":
        return

    def reject(*args, **kwargs):
        raise AssertionError("Network access is prohibited by OSHC_TEST_NO_NETWORK=1")

    monkeypatch.setattr(socket.socket, "connect", reject)
    monkeypatch.setattr(socket, "create_connection", reject)
    import torch.hub
    monkeypatch.setattr(torch.hub, "download_url_to_file", reject)
