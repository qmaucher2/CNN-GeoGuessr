import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import os

class_names = ['AD', 'AE', 'AL', 'AR', 'AS', 'AT', 'AU', 'AX', 'BD', 'BE', 'BG', 'BM', 'BO', 'BR', 'BT', 'BW', 'CA',
               'CC', 'CL', 'CO', 'CW', 'CX', 'CZ', 'DE', 'DK', 'DO', 'EC', 'EE', 'ES', 'FI', 'FO', 'FR', 'GB', 'GH',
               'GI', 'GL', 'GR', 'GT', 'GU', 'HK', 'HR', 'HU', 'ID', 'IE', 'IL', 'IM', 'IN', 'IS', 'IT', 'JE', 'JO',
               'JP', 'KE', 'KG', 'KH', 'KR', 'KZ', 'LA', 'LB', 'LI', 'LK', 'LS', 'LT', 'LU', 'LV', 'MC', 'ME', 'MK',
               'MN', 'MO', 'MP', 'MT', 'MX', 'MY', 'NA', 'NG', 'NL', 'NO', 'NP', 'NZ', 'OM', 'PA', 'PE', 'PH', 'PL',
               'PR', 'PT', 'QA', 'RO', 'RS', 'RU', 'RW', 'SE', 'SG', 'SI', 'SK', 'SM', 'SN', 'ST', 'SZ', 'TH', 'TN',
               'TR', 'TW', 'UA', 'UG', 'US', 'UY', 'VI', 'VN', 'ZA']

class_dict = {
    "Andorra": "AD", "United Arab Emirates": "AE", "Albania": "AL", "Argentina": "AR", "American Samoa": "AS",
    "Austria": "AT", "Australia": "AU", "Aland Islands": "AX", "Bangladesh": "BD", "Belgium": "BE",
    "Bulgaria": "BG", "Bermuda": "BM", "Bolivia": "BO", "Brazil": "BR", "Bhutan": "BT", "Botswana": "BW",
    "Canada": "CA", "Cocos (Keeling) Islands": "CC", "Chile": "CL", "Colombia": "CO", "Costa Rica": "CR",
    "Curaçao": "CW", "Christmas Island": "CX", "Czechia": "CZ", "Germany": "DE", "Denmark": "DK",
    "Dominican Republic": "DO", "Ecuador": "EC", "Estonia": "EE", "Spain": "ES", "Finland": "FI",
    "Faroe Islands": "FO", "France": "FR", "United Kingdom": "GB", "Ghana": "GH", "Gibraltar": "GI",
    "Greenland": "GL", "Greece": "GR", "Guatemala": "GT", "Guam": "GU", "Hong Kong": "HK", "Croatia": "HR",
    "Hungary": "HU", "Indonesia": "ID", "Ireland": "IE", "Israel": "IL", "Isle of Man": "IM", "India": "IN",
    "Iceland": "IS", "Italy": "IT", "Jersey": "JE", "Jordan": "JO", "Japan": "JP", "Kenya": "KE",
    "Kyrgyzstan": "KG", "Cambodia": "KH", "South Korea": "KR", "Kazakhstan": "KZ", "Laos": "LA",
    "Lebanon": "LB", "Liechtenstein": "LI", "Sri Lanka": "LK", "Lesotho": "LS", "Lithuania": "LT",
    "Luxembourg": "LU", "Latvia": "LV", "Monaco": "MC", "Montenegro": "ME", "North Macedonia": "MK",
    "Mongolia": "MN", "Macao": "MO", "Northern Mariana Islands": "MP", "Malta": "MT", "Mexico": "MX",
    "Malaysia": "MY", "Namibia": "NA", "Nigeria": "NG", "Netherlands": "NL", "Norway": "NO", "Nepal": "NP",
    "New Zealand": "NZ", "Oman": "OM", "Panama": "PA", "Peru": "PE", "Philippines": "PH", "Poland": "PL",
    "Puerto Rico": "PR", "Portugal": "PT", "Qatar": "QA", "Romania": "RO", "Serbia": "RS", "Russia": "RU",
    "Rwanda": "RW", "Sweden": "SE", "Singapore": "SG", "Slovenia": "SI", "Slovakia": "SK", "San Marino": "SM",
    "Senegal": "SN", "Sao Tome and Principe": "ST", "Eswatini": "SZ", "Thailand": "TH", "Tunisia": "TN",
    "Turkey": "TR", "Taiwan": "TW", "Ukraine": "UA", "Uganda": "UG", "United States": "US", "Uruguay": "UY",
    "U.S. Virgin Islands": "VI", "Vietnam": "VN", "South Africa": "ZA"
}


def predict_image(model, img_path):
    """
    Processes an image using a relative path passed by the orchestrator.
    """
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = img_array.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)

    preds = []
    pred_vals = []
    for i in np.argsort(predictions[0])[-10:][::-1]:
        preds.append(class_names[i])
        pred_vals.append(predictions[0][i])

    preds2 = [next(k for k, v in class_dict.items() if v == x) for x in preds]

    output_lines = []
    for j, val in zip(preds2, pred_vals):
        output_lines.append(f"{j}: {val:.2%}")

    return output_lines