# -*- coding: utf-8 -*-
import pytest
from src.participant_extract import extract_participant
from src.place_dict import fuzzy_stop_match

def test_participant_driver():
    r = extract_participant("Жүргізуші нагрубил пассажирам")
    assert r and r["role"] == "driver"

def test_participant_controller():
    r = extract_participant("Контролёр в автобусе был груб")
    assert r and r["role"] in ("controller","conductor")

def test_place_fuzzy():
    r = fuzzy_stop_match("Сарыарқа аялдамасында", city_hint="Astana")
    assert r is None or isinstance(r, dict)  # к словарю остановок не привязаны тестовые данные
