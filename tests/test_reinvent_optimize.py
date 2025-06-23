

import logging

from navidiv.reinvent.reinvent_test import reinvent_test

if __name__ == "__main__":
    # Set up logging to see outputs from the class
    logging.basicConfig(level=logging.INFO)

    # Path to your test CSV file (adjust as needed)
    csv_file = "/media/mohammed/Work/Navi_diversity/examples/stage0_1.csv"
    scorer_dicts = [
        {
            "prop_dict": {
                "scorer_name": "Scaffold",
                "scaffold_type": "csk_bm",
                "output_path": "/media/mohammed/Work/Navi_diversity/examples/results/tmp/",
                "min_count_fragments": 1,
                "selectrion_criteria": {
                    "Count_perc_per_molecule": 10,
                    "Count_perc": 0.1,
                    "diff_median_score": 0.1,
                },
            },
            "score_every": 50,
            "groupby_every": 50,
            "selection_criteria": {
                "count_perc_ratio": 5,
                "count_per_molecule": 50,
            },
            "custom_alert_name": "customalertsscaffold",
        },
        {
            "prop_dict": {
                "scorer_name": "Ngram",
                "ngram_size": 10,
                "output_path": "/media/mohammed/Work/Navi_diversity/examples/results/tmp/",
                "min_count_fragments": 5,
                "selectrion_criteria": {
                    "Count_perc_per_molecule": 10,
                    "Count_perc": 0.1,
                    "diff_median_score": 0.1,
                },
            },
            "score_every": 10,
            "groupby_every": 30,
            "selection_criteria": {
                "count_perc_ratio": 10,
                "count_per_molecule": 50,
            },
            "custom_alert_name": "customalertsngrams",
        },
        {
            "prop_dict": {
                "scorer_name": "Cluster",
                "threshold": 0.25,
                "output_path": "/media/mohammed/Work/Navi_diversity/examples/results/tmp/",
                "min_count_fragments": 0,
                "selectrion_criteria": {
                    "Count_perc_per_molecule": 10,
                    "Count_perc": 0.1,
                    "diff_median_score": 0.1,
                },
            },
            "score_every": 10,
            "groupby_every": 30,
            "selection_criteria": {
                "count_perc_ratio": 10,
                "count_per_molecule": 50,
            },
            "custom_alert_name": "dissimilarity",
        },
        {
            "prop_dict": {
                "scorer_name": "Fragments",
                "min_count_fragments": 2,
                "output_path": "/media/mohammed/Work/Navi_diversity/examples/results/tmp/",
                "selectrion_criteria": {
                    "Count_perc_per_molecule": 10,
                    "Count_perc": 0.1,
                    "diff_median_score": 0.1,
                },
            },
            "score_every": 50,
            "groupby_every": 100,
            "selection_criteria": {
                "count_perc_ratio": 10,
                "count_per_molecule": 50,
            },
            "custom_alert_name": "customalerts",
        },
    ]
    # Instantiate the class
    test_instance = reinvent_test(csv_file, scorer_dicts=scorer_dicts)
    # Run the optimize method
    test_instance.optimize()

    print("reinvent_tes.optimize() finished.")
