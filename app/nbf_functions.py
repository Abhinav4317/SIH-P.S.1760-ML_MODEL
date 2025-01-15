# app/nbf_functions.py
import pandas as pd
from app.data_loading import demographics_df, agriculture_df

# SCHEME_WEIGHTS as in the original code
SCHEME_WEIGHTS = {

    # 1. Post Office Savings Account (SB) - General Scheme
    "Post Office Savings Account (SB)": {
        "age_group": {},    # No specific target group
        "gender": {},       
        "occupation": {},   # Open for all
        "income": {}        
    },

    # 2. 5-Year Post Office Recurring Deposit (RD)
    "5-Year Post Office Recurring Deposit (RD)": {
        "age_group": {"0-18" : 0.05 ,"19-35": 0.4, "36-60": 0.45,"60+" : 0.1},  # Working-age individuals
        "income": {"2": 0.3, "3": 0.7},            # Targets mid-income population
        "occupation": {"Salaried Individual": 0.6, "Business Owner": 0.4}
    },

    # 3. Post Office Time Deposit Account (TD)


    # Age Group	Characteristics	Weight
    # 0-18	Limited financial independence; accounts opened by guardians, but less likely to engage actively.	0.05
    # 19-35	Interested in fixed returns for medium-term goals; may prefer TD for its stability.	0.30
    # 36-60	Likely to prioritize security and fixed returns; strong interest in TD accounts for retirement planning.	0.50
    # 60+	Focused on income generation from savings; TD accounts are appealing for regular payouts.	0.15
   
    "Post Office Time Deposit Account (TD)": {
        "age_group": {"0-18":0.05,"19-35": 0.3,"36-60" : 0.5 ,"60+" :0.15},                # Working-age individuals
        "occupation": {"Salaried Individual": 0.6, "Business Owner": 0.3, "Retired" : 0.1},  
        "income": {"2": 0.4, "3": 0.6}              # Mid-income individuals
    },

    
    # 4. Post Office Monthly Income Scheme (MIS)
    "Post Office Monthly Income Scheme (MIS)": {
        "age_group": {"19-35": 0.2,"36-60":0.8},                # Middle-aged individuals
        "income": {"1":0.4,"2":0.3,"3":0.3},             # Mid to higher income
        "occupation": {"Business Owner": 0.3, "Retired": 0.6}
    },

    # 5. Senior Citizen Savings Scheme (SCSS)
    "Senior Citizen Savings Scheme (SCSS)": {
        "age_group": {"60+": 1.0},       # Targets senior citizens
        "occupation": {"Retired": 1.0},  # Retired individuals
        "income": {"2": 0.5, "3": 0.4}   # Middle to low income
    },

    # 6. 15-Year Public Provident Fund (PPF)
    "15-Year Public Provident Fund (PPF)": {
        "age_group": {"19-35": 0.6,"36-60":0.35},                # Younger working-age individuals
        "occupation": {"Salaried Individual": 0.8, "Business Owner": 0.2},
        "income": {"1":0.35, "2":0.25}              # Targets mid to upper income
    },

    # 7. National Savings Certificates (NSC)
    "National Savings Certificates (NSC)": {
        "age_group": {"19-35": 0.4,"36-60" : 0.5},                # Working-age individuals
        "occupation": {"Salaried Individual": 0.6, "Business Owner": 0.3},  
        "income": {"3": 0.7, "4": 0.3}              # Higher income focus
    },

  

    # 8. Sukanya Samriddhi Accounts (SSA)
    "Sukanya Samriddhi Accounts (SSA)": {
        "age_group": {"0-18": 1.0},      # Targets minors
        "gender": {"Female": 1.0},       # Exclusively for female children
        "occupation": {"Student": 0.6},  # Students in rural areas
        "income": {"1": 0.5}             # Low-income families
    },

    # 9. Kisan Vikas Patra (KVP)
    "Kisan Vikas Patra (KVP)": {
        "age_group" : {"0-18":0.05,"36-60":0.7},
        "occupation": {"Farmer": 1.0},  # Exclusively for farmers
        "income": {"2": 0.6, "1": 0.4}, # Low to mid-income rural households
    },

    # 
    "Mahila Samman Savings Certificate": {
          "gender": {"Female": 1.0}, 
          "income": {"2": 0.4,"3":0.25},
          },
}
# Comprehensive Insurance Scheme Mapping with Detailed Weights

SCHEME_WEIGHTS_INSURANCE = {
    'Whole Life Assurance (Suraksha)': {
        "age_group": {
            "0-18": 0.2,    # Limited applicability
            "19-35": 1.0,   # Prime working age, highest potential
            "36-60": 0.8,   # Still strong financial needs
            "60+": 0.3      # Reduced long-term coverage potential
        },
        "gender": {
            "Female": 0.9,  # Slightly higher weight for women's financial security
            "Male": 1.0     # Slightly higher base weight
        },
        "occupation": {
            "Student": 0.3,  # Limited income, low insurance need
            "Retired": 0.5,  # Fixed income, some insurance needs
            "Salaried Individual": 1.0,  # Steady income, prime target
            "Farmer": 0.8,   # Stable but variable income
            "Business Owner": 0.9  # Good financial planning potential
        },
        "income": {
            "1": 0.4,  # Lower income limits coverage
            "2": 0.7,  # Moderate income, good potential
            "3": 1.0,  # Optimal income for comprehensive coverage
            "4": 0.9   # High income, may have alternative investments
        },
        "premium": {
            "Low": 1.0,  # Affordable for most demographics
            "Medium": 0.8,  # Reasonable for mid-income groups
            "High": 0.5,  # Less attractive due to cost
            "Very High": 0.3  # Limited appeal
        }
    },
    
    'Convertible Whole Life Assurance': {
        "age_group": {
            "0-18": 0.3,    # Limited conversion potential
            "19-35": 1.0,   # Maximum flexibility for young adults
            "36-60": 0.7,   # Moderate conversion benefits
            "60+": 0.2      # Very limited conversion potential
        },
        "gender": {
            "Female": 1.0,  # Equal opportunity for financial planning
            "Male": 1.0     # Equal financial instruments
        },
        "occupation": {
            "Student": 0.5,  # Potential for future financial growth
            "Retired": 0.4,  # Limited conversion benefits
            "Salaried Individual": 1.0,  # Ideal for financial flexibility
            "Farmer": 0.7,   # Moderate conversion potential
            "Business Owner": 0.9  # Good financial planning flexibility
        },
        "income": {
            "1": 0.3,  # Limited conversion capability
            "2": 0.6,  # Some financial flexibility
            "3": 1.0,  # Optimal income for conversion
            "4": 0.9   # High income with diverse investment options
        },
        "premium": {
            "Low": 0.7,  # Attractive for budget-conscious
            "Medium": 1.0,  # Sweet spot for convertibility
            "High": 0.6,  # Less attractive
            "Very High": 0.3  # Minimal appeal
        }
    },
    
    'Endowment Assurance (Santosh)': {
        "age_group": {
            "0-18": 0.2,    # No direct earning potential
            "19-35": 0.8,   # Early savings phase
            "36-60": 1.0,   # Peak savings and financial planning
            "60+": 0.5      # Reduced savings potential
        },
        "gender": {
            "Female": 1.0,  # Equal financial planning opportunities
            "Male": 1.0     # Equal savings potential
        },
        "occupation": {
            "Student": 0.3,  # No direct income
            "Retired": 0.6,  # Fixed income savings
            "Salaried Individual": 1.0,  # Consistent savings potential
            "Farmer": 0.7,   # Seasonal income impacts savings
            "Business Owner": 0.9  # Flexible financial planning
        },
        "income": {
            "1": 0.4,  # Limited savings capacity
            "2": 0.7,  # Moderate savings potential
            "3": 1.0,  # Optimal endowment benefits
            "4": 0.9   # High income with diverse investments
        },
        "premium": {
            "Low": 0.6,  # Attractive for budget-conscious
            "Medium": 1.0,  # Ideal savings range
            "High": 0.7,  # Still attractive for serious savers
            "Very High": 0.4  # Limited appeal
        }
    },
    
    'Joint Life Assurance (Yugal Suraksha)': {
        "age_group": {
            "0-18": 0.1,    # Not applicable
            "19-35": 1.0,   # Newly married couples
            "36-60": 0.8,   # Established family units
            "60+": 0.3      # Limited joint life needs
        },
        "gender": {
            "Female": 1.0,  # Equal joint coverage
            "Male": 1.0     # Equal joint protection
        },
        "occupation": {
            "Student": 0.2,  # Not financially established
            "Retired": 0.5,  # Limited joint financial needs
            "Salaried Individual": 1.0,  # Dual income stability
            "Farmer": 0.7,   # Combined rural income strategy
            "Business Owner": 0.9  # Complementary financial protection
        },
        "income": {
            "1": 0.4,  # Limited joint resources
            "2": 0.7,  # Moderate joint financial planning
            "3": 1.0,  # Optimal joint coverage
            "4": 0.9   # High income with complex financial needs
        },
        "premium": {
            "Low": 0.6,  # Attractive for young couples
            "Medium": 1.0,  # Ideal joint coverage
            "High": 0.7,  # Still reasonable for comprehensive protection
            "Very High": 0.4  # Less appealing
        }
    },
    
    'Anticipated Endowment Assurance': {
        "age_group": {
            "0-18": 0.2,    # No direct financial planning
            "19-35": 0.8,   # Early financial goal setting
            "36-60": 1.0,   # Peak mid-term financial planning
            "60+": 0.5      # Limited mid-term goal potential
        },
        "gender": {
            "Female": 1.0,  # Equal financial opportunity
            "Male": 1.0     # Equal periodic payout potential
        },
        "occupation": {
            "Student": 0.3,  # No direct income
            "Retired": 0.6,  # Fixed income considerations
            "Salaried Individual": 1.0,  # Consistent periodic planning
            "Farmer": 1.0,   # Matches seasonal income cycles
            "Business Owner": 0.9  # Flexible financial management
        },
        "income": {
            "1": 0.4,  # Limited periodic payout benefits
            "2": 0.7,  # Moderate financial flexibility
            "3": 1.0,  # Optimal periodic payout potential
            "4": 0.9   # High income with diverse investment strategies
        },
        "premium": {
            "Low": 0.6,  # Attractive for budget-conscious
            "Medium": 1.0,  # Ideal periodic savings
            "High": 0.7,  # Still reasonable for structured savings
            "Very High": 0.4  # Limited appeal
        }
    },
    
    'Whole Life Assurance (Gram Suraksha)': {
        "age_group": {
            "0-18": 0.2,    # Limited rural economic participation
            "19-35": 1.0,   # Peak rural working age
            "36-60": 0.8,   # Continued rural economic activity
            "60+": 0.4      # Reduced economic potential
        },
        "gender": {
            "Female": 0.9,  # Slightly higher rural economic vulnerability
            "Male": 1.0     # Traditional rural economic lead
        },
        "occupation": {
            "Student": 0.3,  # No direct rural income
            "Retired": 0.5,  # Limited rural economic activity
            "Salaried Individual": 0.7,  # Limited urban-rural employment
            "Farmer": 1.0,   # Primary target demographic
            "Business Owner": 0.8  # Rural entrepreneurial potential
        },
        "income": {
            "1": 1.0,  # Specifically designed for low-income rural population
            "2": 0.7,  # Transitional rural income group
            "3": 0.4,  # Less targeted for higher rural incomes
            "4": 0.2   # Minimal rural high-income representation
        },
        "premium": {
            "Low": 1.0,  # Highly affordable for rural populations
            "Medium": 0.7,  # Reasonable for rural economic conditions
            "High": 0.4,  # Limited rural affordability
            "Very High": 0.2  # Minimal rural appeal
        }
    },
    
    'Convertible Whole Life Assurance (Gram Suvidha)': {
        "age_group": {
            "0-18": 0.3,    # Limited rural conversion potential
            "19-35": 1.0,   # Maximum rural financial flexibility
            "36-60": 0.7,   # Moderate rural conversion benefits
            "60+": 0.3      # Very limited rural conversion potential
        },
        "gender": {
            "Female": 0.9,  # Empowerment through financial flexibility
            "Male": 1.0     # Traditional rural financial instrument
        },
        "occupation": {
            "Student": 0.4,  # Future rural economic potential
            "Retired": 0.5,  # Limited rural conversion benefits
            "Salaried Individual": 0.7,  # Limited urban-rural employment
            "Farmer": 1.0,   # Ideal rural financial flexibility
            "Business Owner": 0.9  # Rural entrepreneurial adaptability
        },
        "income": {
            "1": 1.0,  # Specifically affordable for low-income rural population
            "2": 0.7,  # Moderate rural income group
            "3": 0.4,  # Less targeted for higher rural incomes
            "4": 0.2   # Minimal rural high-income representation
        },
        "premium": {
            "Low": 1.0,  # Highly attractive for rural budget
            "Medium": 0.7,  # Reasonable rural conversion potential
            "High": 0.4,  # Limited rural affordability
            "Very High": 0.2  # Minimal rural appeal
        }
    },
    
    'Endowment Assurance (Gram Santosh)': {
        "age_group": {
            "0-18": 0.2,    # No direct rural economic participation
            "19-35": 0.8,   # Early rural savings potential
            "36-60": 1.0,   # Peak rural savings and financial planning
            "60+": 0.5      # Reduced rural savings potential
        },
        "gender": {
            "Female": 0.9,  # Rural economic empowerment
            "Male": 1.0     # Traditional rural financial lead
        },
        "occupation": {
            "Student": 0.3,  # No direct rural income
            "Retired": 0.6,  # Fixed rural income savings
            "Salaried Individual": 0.7,  # Limited urban-rural employment
            "Farmer": 1.0,   # Seasonal income matches endowment
            "Business Owner": 0.8  # Rural entrepreneurial savings
        },
        "income": {
            "1": 0.8,  # Affordable savings for low-income rural population
            "2": 1.0,  # Optimal rural income for endowment
            "3": 0.6,  # Less targeted for higher rural incomes
            "4": 0.3   # Minimal rural high-income representation
        },
        "premium": {
            "Low": 1.0,  # Highly attractive for rural savings
            "Medium": 0.8,  # Reasonable rural endowment potential
            "High": 0.5,  # Limited rural affordability
            "Very High": 0.3  # Minimal rural appeal
        }
    },
    
        'Anticipated Endowment Assurance (Gram Sumangal)': {
        "age_group": {
            "0-18": 0.2,    # No direct rural economic participation
            "19-35": 0.8,   # Early rural financial goal setting
            "36-60": 1.0,   # Peak rural mid-term financial planning
            "60+": 0.5      # Limited rural mid-term goal potential
        },
        "gender": {
            "Female": 0.9,  # Rural economic empowerment
            "Male": 1.0     # Traditional rural financial lead
        },
        "occupation": {
            "Student": 0.3,  # No direct rural income
            "Retired": 0.6,  # Fixed rural income considerations
            "Salaried Individual": 0.7,  # Limited urban-rural employment
            "Farmer": 1.0,   # Perfectly matches seasonal income cycles
            "Business Owner": 0.8  # Rural entrepreneurial flexibility
        },
        "income": {
            "1": 0.7,  # Supports low-income periodic needs
            "2": 1.0,  # Optimal periodic payout for rural income
            "3": 0.6,  # Less targeted for higher rural incomes
            "4": 0.3   # Minimal rural high-income representation
        },
        "premium": {
            "Low": 1.0,  # Highly attractive for rural periodic savings
            "Medium": 0.8,  # Reasonable rural periodic potential
            "High": 0.5,  # Limited rural affordability
            "Very High": 0.3  # Minimal rural appeal
        }
    }
}



def calculate_demographic_weight(post_office_name, scheme_name, is_insurance=False):
    po_data = demographics_df[demographics_df["Post Office Name"] == post_office_name]
    weights = (
        SCHEME_WEIGHTS.get(scheme_name, {})
        if not is_insurance
        else SCHEME_WEIGHTS_INSURANCE.get(scheme_name, {})
    )
    score = 0

    for _, row in po_data.iterrows():
        age_wt = weights.get("age_group", {}).get(row["Age Group"], 0)
        gender_wt = weights.get("gender", {}).get(row["Gender"], 0)
        occ_wt = weights.get("occupation", {}).get(row["Occupation"], 0)
        income_wt = weights.get("income", {}).get(str(row["Income Level"]), 0)

        # Contribution of each demographic segment
        score += row["Population"] * (age_wt + gender_wt + occ_wt + income_wt)
    return score


def calculate_agriculture_weight(district, current_month):
    crop_data = agriculture_df[agriculture_df["District"] == district]
    score = 0

    for _, row in crop_data.iterrows():
        sowing_active = (
            row["Sowing Period Numeric"]
            <= current_month
            <= row["Harvesting Period Numeric"]
        )
        harvesting_active = current_month == row["Harvesting Period Numeric"]
        if sowing_active:
            score += 0.6  # Sowing weight
        if harvesting_active:
            score += 0.4  # Harvesting weight
    return score


def calculate_nbf(post_office_name, current_month, is_insurance=False):
    nbf_scores = {}
    po_data = demographics_df[demographics_df["Post Office Name"] == post_office_name]
    if po_data.empty:
        raise ValueError("Post office not found in demographics data.")
    district = po_data["District"].iloc[0]

    selected_scheme_weights = (
        SCHEME_WEIGHTS if not is_insurance else SCHEME_WEIGHTS_INSURANCE
    )

    for scheme_name in selected_scheme_weights.keys():
        # Calculate weights
        demo_weight = calculate_demographic_weight(
            post_office_name, scheme_name, is_insurance
        )
        agri_weight = calculate_agriculture_weight(district, current_month)

        # Combine weights with coefficients
        alpha, beta = 0.7, 0.3
        nbf_scores[scheme_name] = alpha * demo_weight + beta * agri_weight

    # Return NBF scores as a Series
    return pd.Series(nbf_scores)
