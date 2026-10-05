.PHONY: install install-reports test smoke smoke-pa smoke-disorder smoke-anderson inventory figures validate

install:
	python3 -m pip install -e .

install-reports:
	python3 -m pip install -e '.[reports]'

test:
	python3 -m unittest discover -s tests -v

smoke:
	python3 -m small_world_mobility_edge.model_smoke --model all --n 160 --degree 6 --tau 2.5 --num-graphs 3 --output-dir outputs/smoke

smoke-pa:
	python3 -m small_world_mobility_edge.preferential_attachment.experiments --smoke-test --output-dir outputs/smoke/pa

smoke-disorder:
	python3 tools/run_pa_diagonal_uniform_noise_smoke.py --output-dir outputs/smoke/pa_diagonal_uniform_noise

smoke-anderson:
	python3 tools/run_anderson_rrg_smoke.py --output-dir outputs/smoke/anderson_rrg

inventory:
	python3 tools/build_source_inventory.py

figures:
	python3 tools/audit_report_figures.py --write

validate: test inventory figures
	python3 tools/validate_release.py
