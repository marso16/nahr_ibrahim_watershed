python src\data\split.py

python src\data\windowing.py --horizon 1 --lookback 60
python src\data\windowing.py --horizon 3 --lookback 60
python src\data\windowing.py --horizon 14 --lookback 90

python src\data\build_sequences_hybrid.py --horizon 1 --lookback 60
python src\data\build_sequences_hybrid.py --horizon 3 --lookback 60
python src\data\build_sequences_hybrid.py --horizon 14 --lookback 90