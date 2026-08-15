from actions.smart_data_scientist import CarlDataEngine

# Set to your new CSV file name
file_name = "german_credit_data.csv"

try:
    print("==================================================")
    print("        PROJECT CARL DYNAMIC DATA ENGINE TEST     ")
    print("==================================================\n")

    engine = CarlDataEngine(file_name)

    # 1. DYNAMIC DETAILED DATA AUDIT
    print("--- 1. DETAILED DATASET BREAKDOWN ---")
    report = engine.get_detailed_report()
    print(f"File Name: {report['file_name']}")
    print(f"Total Rows: {report['total_rows']}")
    print(f"Total Columns: {report['total_columns']}")
    print(f"Ignored Index/ID Columns: {report['ignored_id_columns']}")
    print(f"Auto-Detected Target Column: {report['auto_detected_target']}\n")

    print("--- COLUMN-BY-COLUMN DETAILS ---")
    for col in report['column_breakdown']:
        print(f"• {col['column_name']} [{col['data_type']}] | Missing: {col['missing_values']} | Unique: {col['unique_values']}")
        if "mean" in col:
            print(f"   -> Min: {col['min']} | Max: {col['max']} | Mean: {col['mean']}")
        if "top_categories" in col:
            print(f"   -> Top Categories: {col['top_categories']}")
    print("\n")

    # 2. INTERACTIVE CHART GENERATION
    print("--- 2. PLOTLY INTERACTIVE CHART TEST ---")
    first_col = report['column_breakdown'][0]['column_name']
    second_col = report['column_breakdown'][1]['column_name']
    chart_json = engine.generate_chart("histogram", first_col)
    print(f"Generated histogram JSON for '{first_col}' (Length: {len(chart_json)} chars)\n")

    # 3. AUTOMATED MACHINE LEARNING PREDICTION
    print("--- 3. AUTOMATED PREDICTION ENGINE TEST ---")
    # You can pass any target column, e.g., 'Purpose', 'Housing', or let it auto-detect!
    target = "Purpose" 
    prediction_result = engine.predict_target(target)
    print(f"Target Column Predicted: {prediction_result.get('target')}")
    print(f"Machine Learning Task: {prediction_result.get('task_type')}")
    print(f"Model Accuracy Score: {prediction_result.get('accuracy_score')}")
    print(f"Top 5 Key Drivers: {prediction_result.get('top_key_factors')}")

except Exception as e:
    print("Execution Error:", e)