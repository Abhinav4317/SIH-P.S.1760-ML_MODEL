import pandas as pd


def calculate_projections(district_name: str, data: pd.DataFrame, growth_rates: dict) -> dict:
    """
    Generates 5-year projections for workforce trends based on growth rates.
    """
    district_filtered = data[data["Area_Name"] == district_name]
    if district_filtered.empty:
        raise ValueError(f"No data found for district: {district_name}")

    # Extract initial values
    initial_values = {
        "Workforce_Participation": district_filtered["Workforce_Participation_Rate (%)"].iloc[0],
        "Projected_Workforce": district_filtered["Projected_Workforce_Persons (5 Years)"].iloc[0],
        "Elderly_Workers": district_filtered["Elderly_Workers_Projected"].iloc[0],
        "Urban_Workforce": district_filtered["Projected_Urban_Workforce (5 Years)"].iloc[0],
        "Female_Workforce_Inclusion": district_filtered["Projected_Female_Workforce_Inclusion"].iloc[0],
    }

    # Generate projections for the next 5 years
    projections = {key: [] for key in initial_values.keys()}
    for i in range(5):  # Next 5 years
        projections["Workforce_Participation"].append(
            initial_values["Workforce_Participation"] * (1 + growth_rates["Workforce_Participation"]) ** i
        )
        projections["Projected_Workforce"].append(
            initial_values["Projected_Workforce"] * (1 + growth_rates["Workforce_Growth"]) ** i
        )
        projections["Elderly_Workers"].append(
            initial_values["Elderly_Workers"] * (1 + growth_rates["Elderly_Workers"]) ** i
        )
        projections["Urban_Workforce"].append(
            initial_values["Urban_Workforce"] * (1 + growth_rates["Urbanization"]) ** i
        )
        projections["Female_Workforce_Inclusion"].append(
            initial_values["Female_Workforce_Inclusion"] * (1 + growth_rates["Female_Literacy"]) ** i
        )

    return projections