import pytest

from siar_forecast_raw.errors import SensitiveHtmlError
from siar_forecast_raw.secret_scan import assert_html_safe, find_reusable_session_values


def test_blocks_reusable_csrf_and_session_values():
    html = b'<input type="hidden" name="_csrf" value="a-very-long-reusable-token-123">'
    assert find_reusable_session_values(html)
    with pytest.raises(SensitiveHtmlError):
        assert_html_safe(html)


@pytest.mark.parametrize("html", [
    b'<input type="hidden" name="_csrf" value="">',
    b'<meta name="csrf-token" content="">',
    b'<div>_csrf Cookie XSRF-TOKEN JSESSIONID Authorization</div>',
    b'<div id="forecast-row-1234567890">Weather forecast</div>',
    b'<script src="/assets/app.0123456789abcdef0123456789abcdef.js"></script>',
    b'<p>ordinary forecast token unit</p>',
])
def test_allows_empty_markers_and_ordinary_values(html):
    assert find_reusable_session_values(html) == []
    assert_html_safe(html)


def test_known_session_value_is_blocked_without_sensitive_name():
    assert find_reusable_session_values(b'<span>abcDEF0123456789</span>', {"abcDEF0123456789"})


def test_blocks_unquoted_reusable_query_token():
    assert find_reusable_session_values(b'<a href="/page?access_token=abcDEF0123456789">forecast</a>')
