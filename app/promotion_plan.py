from app.data_loading import client, post_office_schemes, final_df
from app.models import PromotionPlan
from app.utils import collate_predictions, get_demographics
import pandas as pd


def generate_promotion_plan(scheme_name, scheme_details, demographics_data):
    user_message = f"""
    Scheme Name: {scheme_name}

    Scheme Details: {scheme_details}

    Demographics and Agriculture Data:
    {demographics_data}
    """

    system_message = """
    You are a strategic marketing assistant specializing in developing promotional plans for financial schemes in rural and semi-urban postal regions. Your goal is to provide a detailed and actionable plan to promote a specific scheme based on its details and the demographic, agricultural, and socio-economic data of the region.

    - Focus on tailoring the plan to the unique characteristics of the region.
    - Emphasize strategies that are practical, culturally relevant, and aligned with the demographics and economic activities of the population.
    - Provide clear steps, including communication channels, collaboration opportunities, and incentives to boost adoption of the scheme.
    - Highlight how the scheme benefits the specific target groups in the region.

    Output the plan in the following structure:
    1. **Scheme Overview**: A brief introduction to the scheme.
    2. **Target Audience**: Identify specific groups (e.g., farmers, salaried individuals, retired persons) who will benefit most.
    3. **Promotion Strategies**: Detailed steps on how to promote the scheme, including:
       - Communication channels (e.g., community meetings, SMS campaigns, pamphlets).
       - Partnerships (e.g., local agricultural cooperatives, schools, or businesses).
       - Incentives or highlights of the scheme's benefits.
    4. **Execution Timeline**: A high-level timeline to roll out the promotion plan.
    5. **Key Metrics**: How success will be measured (e.g., number of enrollments, awareness levels, feedback).
    """

    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )

    generated_plan = completion.choices[0].message.content
    return PromotionPlan(scheme_name=scheme_name, plan=generated_plan)


def collate_and_generate_plan(
    post_office_name,
    model1,
    model2,
    x_df,
    final_df,
    numeric_cols_1,
    numeric_cols_2,
    months=23,
    month_offset=1,
    top_n_schemes=3,
    include_neighbor_vote=True,
):
    top_schemes = collate_predictions(
        post_office_name,
        model1,
        model2,
        x_df,
        final_df,
        numeric_cols_1,
        numeric_cols_2,
        months,
        month_offset,
        top_n_schemes,
        include_neighbor_vote,
    )
    plan_list = []
    for scheme in top_schemes:
        scheme_details = post_office_schemes[scheme]
        demographics_data = get_demographics(post_office_name)
        plan = generate_promotion_plan(scheme, scheme_details, demographics_data)
        plan_list.append(plan)
    return plan_list