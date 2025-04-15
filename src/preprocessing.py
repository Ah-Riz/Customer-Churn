import pandas as pd

def preprocessing(path, gender_list, MultipleLines_list, InternetService_list, service_installation, Contract_list, paymentMethod_list):
    raw_data = pd.read_csv(path)
    clean_data = processing_data(raw_data, gender_list, MultipleLines_list, InternetService_list, service_installation, Contract_list, paymentMethod_list)
    return clean_data

def processing_data(raw_data, gender_list, MultipleLines_list, InternetService_list, service_installation, Contract_list, paymentMethod_list):
    row = {}
    row["id_result"] = check_customerID(raw_data["customerID"])
    row["gender_result"] = check_gender(raw_data["gender"], gender_list.keys())
    row["seniorCitizen_result"] = check_SeniorCitizen(raw_data["SeniorCitizen"])
    row["partner_result"] = check_Partner(raw_data["Partner"])
    row["dependents_result"] = check_Dependents(raw_data["Dependents"])
    row["tenure_result"] = check_tenure(raw_data["tenure"])
    row["phoneService_result"] = check_PhoneService(raw_data["PhoneService"])
    row["multipleLines_result"] = check_MultipleLines(raw_data["MultipleLines"], MultipleLines_list.keys())
    row["internerService_result"] = check_InternetService(raw_data["InternetService"], InternetService_list.keys())
    row["onlineSecurity_result"] = check_Service(raw_data["OnlineSecurity"], service_installation.keys())
    row["onlineBackup_result"] = check_Service(raw_data["OnlineBackup"], service_installation.keys())
    row["deviceProtection_result"] = check_Service(raw_data["DeviceProtection"], service_installation.keys())
    row["techSupport_result"] = check_Service(raw_data["TechSupport"], service_installation.keys())
    row["streamingTV_result"] = check_Service(raw_data["StreamingTV"], service_installation.keys())
    row["stramingMovies_result"] = check_Service(raw_data["StreamingMovies"], service_installation.keys())
    row["contract_result"] = check_Contract(raw_data["Contract"], Contract_list.keys())
    row["paperlessBilling_result"] = check_PaperlessBilling(raw_data["PaperlessBilling"])
    row["paymentMethod_result"] = check_PaymentMethod(raw_data["PaymentMethod"], paymentMethod_list.keys())
    row["monthlyCharges_result"] = check_Charges(raw_data["MonthlyCharges"])
    row["totalCharges_result"] = check_Charges(raw_data["TotalCharges"])
    row["churn_result"] = check_Churn(raw_data["Churn"])

    clean_data = delete_rows(raw_data, row)

    clean_data["gender"] = clean_data["gender"].str.lower().map(gender_list)
    clean_data["Partner"] = clean_data["Partner"].str.lower().map({"yes": 1, "no": 0})
    clean_data["Dependents"] = clean_data["Dependents"].str.lower().map({"yes": 1, "no": 0})
    clean_data["PhoneService"] = clean_data["PhoneService"].str.lower().map({"yes": 1, "no": 0})
    clean_data["MultipleLines"] = clean_data["MultipleLines"].str.lower().map(MultipleLines_list)
    clean_data["InternetService"] = clean_data["InternetService"].str.lower().map(InternetService_list)
    clean_data["OnlineSecurity"] = clean_data["OnlineSecurity"].str.lower().map(service_installation)
    clean_data["OnlineBackup"] = clean_data["OnlineBackup"].str.lower().map(service_installation)
    clean_data["DeviceProtection"] = clean_data["DeviceProtection"].str.lower().map(service_installation)
    clean_data["TechSupport"] = clean_data["TechSupport"].str.lower().map(service_installation)
    clean_data["StreamingTV"] = clean_data["StreamingTV"].str.lower().map(service_installation)
    clean_data["StreamingMovies"] = clean_data["StreamingMovies"].str.lower().map(service_installation)
    clean_data["Contract"] = clean_data["Contract"].str.lower().map(Contract_list)
    clean_data["PaperlessBilling"] = clean_data["PaperlessBilling"].str.lower().map({"yes": 1, "no": 0})
    clean_data["PaymentMethod"] = clean_data["PaymentMethod"].str.lower().map(paymentMethod_list)
    clean_data["Churn"] = clean_data["Churn"].str.lower().map({"yes": 1, "no": 0})

    return clean_data

def check_customerID(customer_id):
    null_indices = customer_id[customer_id.isna()].index
    duplicate_indices = customer_id[customer_id.duplicated(keep=False)].index

    if not null_indices.empty or not duplicate_indices.empty:
        result = {}
        if not null_indices.empty:
            result["null_rows"] = null_indices  
        if not duplicate_indices.empty:
            result["duplicate_rows"] = duplicate_indices
        return result
    else:
        return None

def check_gender(gender, gender_list):
    gender = gender.str.lower()
    invalid_indices = gender[~gender.isin(gender_list)].index

    if not invalid_indices.empty:
        result = {}
        result["invalid_indices"] = invalid_indices
        return result
    else:
        return None

def check_SeniorCitizen(senior_citizen):
    non_binary_indices = senior_citizen[~senior_citizen.isin([0, 1])].index

    if not non_binary_indices.empty:
        result = {}
        result["non_binary_indices"] = non_binary_indices
        return result
    else:
        return None

def check_Partner(partner):
    partner = partner.str.lower()
    out_of_range_indices = partner[~partner.isin(["yes", "no"])].index
    if not out_of_range_indices.empty:
        result = {}
        result["out_of_range_indices"] = out_of_range_indices
        return result
    else:
        return None

def check_Dependents(dependents):
    dependents = dependents.str.lower()
    out_of_range_indices = dependents[~dependents.isin(["yes", "no"])].index
    if not out_of_range_indices.empty:
        result = {}
        result["out_of_range_indices"] = out_of_range_indices
        return result
    else:
        return None

def check_tenure(tenure):
    null_indices = tenure[tenure.isna()].index
    negative_indices = tenure[tenure < 0].index

    if not null_indices.empty or not negative_indices.empty:
        result = {}
        if not null_indices.empty:
            result["null_indices"] = null_indices
        if not negative_indices.empty:
            result["negative_indices"] = negative_indices
        return result
    else:
        return None

def check_PhoneService(phone_service):
    phone_service = phone_service.str.lower()
    out_of_range_indices = phone_service[~phone_service.isin(["yes", "no"])].index
    
    if not out_of_range_indices.empty:
        result = {}
        result["out_of_range_indices"] = out_of_range_indices
        return result
    else:
        return None

def check_MultipleLines(multiple_lines, multiple_lines_list):
    multiple_lines = multiple_lines.str.lower()
    out_of_range_indices = multiple_lines[~multiple_lines.isin(multiple_lines_list)].index
    if not out_of_range_indices.empty:
        result = {}
        result["out_of_range_indices"] = out_of_range_indices
        return result
    else:
        return None

def check_InternetService(internet_service, InternetService_list):
    internet_service = internet_service.str.lower()
    out_of_range_indices = internet_service[~internet_service.isin(InternetService_list)].index
    if not out_of_range_indices.empty:
        result = {}
        result["out_of_range_indices"] = out_of_range_indices
        return result
    else:   
        return None

def check_Service(service, list):
    service = service.str.lower()
    out_of_range_indices = service[~service.isin(list)].index

    if not out_of_range_indices.empty:
        result = {}
        result["out_of_range_indices"] = out_of_range_indices
        return result
    else:
        return None

def check_Contract(contract, Contract_list):
    contract = contract.str.lower()
    out_of_range_indices = contract[~contract.isin(Contract_list)].index
    
    if not out_of_range_indices.empty:
        result = {}
        result["out_of_range_indices"] = out_of_range_indices
        return result
    else:
        return None

def check_PaperlessBilling(paperless_billing):
    paperless_billing = paperless_billing.str.lower()
    out_of_range_indices = paperless_billing[~paperless_billing.isin(["yes", "no"])].index

    if not out_of_range_indices.empty:
        result = {}
        result["out_of_range_indices"] = out_of_range_indices
        return result
    else:
        return None

def check_PaymentMethod(payment_method, list):
    payment_method = payment_method.str.lower()
    out_of_range_indices = payment_method[~payment_method.isin(list)].index
    
    if not out_of_range_indices.empty:
        result = {}
        result["out_of_range_indices"] = out_of_range_indices
        return result
    else:
        return None

def check_Charges(charges):
    null_indices = charges[charges.isna()].index
    negative_indices = charges[charges < 0].index

    if not null_indices.empty or not negative_indices.empty:
    # if not negative_indices.empty:
        result = {}
        if not null_indices.empty:
            result["null_indices"] = null_indices
        if not negative_indices.empty:
            result["negative_indices"] = negative_indices
        return result
    else:
        return None

def check_Churn(churn):
    churn = churn.str.lower()
    out_of_range_indices = churn[~churn.isin(["yes", "no"])].index

    if not out_of_range_indices.empty:
        result = {}
        result["out_of_range_indices"] = out_of_range_indices
        return result
    else:
        return None
    
def delete_rows(raw_data, row):
    clean_row = check_duplicate_rows(row)
    raw_data.drop(clean_row, inplace=True)
    return raw_data

def check_duplicate_rows(row):
    indices = []
    for key, value in row.items():
        if value is not None:
            for key2,value2 in row[key].items():
                indices.extend(value2)
    set_indices = list(set(indices))
    return set_indices

if __name__ == "__main__":
    path = "data/test.csv"
    gender_list = {"female":0, "male":1}
    MultipleLines_list = {"yes": 1, "no": 0, "no phone service": -1}
    service_installation = {"yes": 1, "no": 0, "no internet service": -1}
    InternetService_list = {"dsl": 1, "fiber optic": 2, "no": 0}
    Contract_list = {"month-to-month": 0, "one year": 1, "two year": 2}
    paymentMethod_list = {"bank transfer (automatic)": 0, "credit card (automatic)": 1, "electronic check": 2, "mailed check": 3}
    preprocessing(path, gender_list, MultipleLines_list, InternetService_list, service_installation, Contract_list, paymentMethod_list) 