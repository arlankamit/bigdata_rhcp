import re
from src.extractors import extract_route, extract_place, extract_time

def test_route_ru():
    assert extract_route("маршрут 12 опоздал") == "12"
    assert extract_route("автобусы 128 не остановились") == "128"

def test_place_ru_kk():
    assert extract_place("на остановке Сарыарка в 08:30") == "Сарыарка"
    assert extract_place("Сарыарқа аялдамасына 09:10 уақытында келмеді") == "Сарыарқа"

def test_time():
    assert extract_time("в 08:30") == "08:30"
