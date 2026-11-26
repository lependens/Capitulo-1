import pytest
from aemet_forecast_collector.parsing import canonical_municipality_id, summarize_payload, PayloadContractError

@pytest.mark.parametrize('value', [7040, '07040'])
def test_palma_identifier(value):
    assert canonical_municipality_id(value) == '07040'

@pytest.mark.parametrize('value', [True, False, 7040.0, 7040.1, '7040', '07040 ', '０７０４０', None, -1, 100000, '00000', 53001, 7000, [], {}])
def test_invalid_identifier(value):
    with pytest.raises(PayloadContractError):
        canonical_municipality_id(value)

@pytest.mark.parametrize('value', [7041, '07041'])
def test_other_municipality_rejected(value):
    with pytest.raises(PayloadContractError, match='does not match'):
        summarize_payload([{'id': value}], '07040')
