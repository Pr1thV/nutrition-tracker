"""
Offline Agent Evaluation Suite.
Measures Calorie MAPE, Protein MAPE, and Semantic Grounding Accuracy against Golden Benchmark.
"""
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from agent.tools.lookup_nutrition_tool import lookup_nutrition_profile
from db.connection import AsyncSessionLocal, init_db
from db.seed_ifct import seed_ifct_database

logger = logging.getLogger(__name__)

BENCHMARK_PATH = Path(__file__).resolve().parent / "golden_tests" / "test_cases.json"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"


async def run_agent_evaluations() -> Dict[str, Any]:
    """
    Executes the golden benchmark evaluation suite.
    """
    await init_db()
    await seed_ifct_database()

    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    test_cases: List[Dict[str, Any]] = bench_data.get("test_cases", [])
    logger.info(f"Loaded {len(test_cases)} golden test cases from {BENCHMARK_PATH}")

    calorie_errors = []
    protein_errors = []
    grounded_count = 0
    total_items = 0

    detailed_results = []

    async with AsyncSessionLocal() as session:
        for tc in test_cases:
            tc_id = tc["id"]
            desc = tc["description"]
            expected = tc["expected"]

            if tc.get("query_type") == "multi_item_thali":
                items = tc.get("items", [])
                total_cal = 0.0
                total_pro = 0.0
                for itm in items:
                    total_items += 1
                    nutr = await lookup_nutrition_profile(
                        food_name=itm["name"],
                        portion_grams=itm["portion_grams"],
                        session=session,
                    )
                    total_cal += nutr["calories"]
                    total_pro += nutr["protein"]
                    if nutr.get("is_grounded_in_ifct"):
                        grounded_count += 1
            else:
                total_items += 1
                nutr = await lookup_nutrition_profile(
                    food_name=tc["food_name"],
                    portion_grams=tc["portion_grams"],
                    session=session,
                )
                total_cal = nutr["calories"]
                total_pro = nutr["protein"]
                if nutr.get("is_grounded_in_ifct"):
                    grounded_count += 1

            # Percentage Errors
            cal_err = abs(total_cal - expected["calories"]) / expected["calories"]
            pro_err = abs(total_pro - expected["protein_g"]) / expected["protein_g"]

            calorie_errors.append(cal_err)
            protein_errors.append(pro_err)

            detailed_results.append(
                {
                    "test_case_id": tc_id,
                    "description": desc,
                    "expected_cal": expected["calories"],
                    "actual_cal": round(total_cal, 1),
                    "cal_error_pct": f"{cal_err * 100:.1f}%",
                    "expected_protein": expected["protein_g"],
                    "actual_protein": round(total_pro, 1),
                    "protein_error_pct": f"{pro_err * 100:.1f}%",
                }
            )

    # Calculate Aggregate Metrics
    calorie_mape = float(sum(calorie_errors) / len(calorie_errors)) * 100.0
    protein_mape = float(sum(protein_errors) / len(protein_errors)) * 100.0
    grounding_rate = float(grounded_count / total_items) * 100.0

    eval_summary = {
        "benchmark_name": bench_data.get("benchmark_name"),
        "total_test_cases": len(test_cases),
        "total_food_items_evaluated": total_items,
        "metrics": {
            "calorie_mape_pct": round(calorie_mape, 2),
            "protein_mape_pct": round(protein_mape, 2),
            "ifct_grounding_accuracy_pct": round(grounding_rate, 2),
            "pass_status": "PASSED" if calorie_mape < 10.0 and grounding_rate >= 90.0 else "WARNING",
        },
        "detailed_results": detailed_results,
    }

    # Print Report
    print("\n" + "=" * 55)
    print("      AGENT NUTRITION ESTIMATION EVALUATION REPORT")
    print("=" * 55)
    print(f"  Calorie MAPE (Error)........... {calorie_mape:.2f}% (Threshold < 10%)")
    print(f"  Protein MAPE (Error)........... {protein_mape:.2f}% (Threshold < 10%)")
    print(f"  IFCT Grounding Rate............ {grounding_rate:.1f}% (Threshold >= 90%)")
    print(f"  Overall Status................. {eval_summary['metrics']['pass_status']}")
    print("=" * 55 + "\n")

    # Save to reports directory
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = REPORTS_DIR / "agent_eval_summary.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    return eval_summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_agent_evaluations())
