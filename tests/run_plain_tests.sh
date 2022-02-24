# plain run is deprecated
# for f in tests/test_*.py; do python "$f"; done
pytest --ignore="tests/test_clustering.py" -v