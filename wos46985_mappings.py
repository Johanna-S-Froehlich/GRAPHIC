"""Class metadata for WOS-46985 (134-class Web of Science).

Exposes what ``metrics.py`` looks up:
  - ``class_codes``:    the 134 class codes, in confusion-matrix order. The
                        pipelines sort the codes as strings, so ``class_10``
                        comes right after ``class_1``.
  - ``fine_labels``:    the class names, in the same order. Some class indices
                        carry more than one name in the dataset; the most
                        frequent one is used.
  - ``human_grouping``: the human baseline grouping of the classes (25 groups)
  - ``mappings``:       the same grouping in the form used for assortativity
"""

class_codes = [
    "class_0", "class_1", "class_10", "class_100", "class_101", "class_102",
    "class_103", "class_104", "class_105", "class_106", "class_107", "class_108",
    "class_109", "class_11", "class_110", "class_111", "class_112", "class_113",
    "class_114", "class_115", "class_116", "class_117", "class_118", "class_119",
    "class_12", "class_120", "class_121", "class_122", "class_123", "class_124",
    "class_125", "class_126", "class_127", "class_128", "class_129", "class_13",
    "class_130", "class_131", "class_132", "class_133", "class_14", "class_15",
    "class_16", "class_17", "class_18", "class_19", "class_2", "class_20", "class_21",
    "class_22", "class_23", "class_24", "class_25", "class_26", "class_27", "class_28",
    "class_29", "class_3", "class_30", "class_31", "class_32", "class_33", "class_34",
    "class_35", "class_36", "class_37", "class_38", "class_39", "class_4", "class_40",
    "class_41", "class_42", "class_43", "class_44", "class_45", "class_46", "class_47",
    "class_48", "class_49", "class_5", "class_50", "class_51", "class_52", "class_53",
    "class_54", "class_55", "class_56", "class_57", "class_58", "class_59", "class_6",
    "class_60", "class_61", "class_62", "class_63", "class_64", "class_65", "class_66",
    "class_67", "class_68", "class_69", "class_7", "class_70", "class_71", "class_72",
    "class_73", "class_74", "class_75", "class_76", "class_77", "class_78", "class_79",
    "class_8", "class_80", "class_81", "class_82", "class_83", "class_84", "class_85",
    "class_86", "class_87", "class_88", "class_89", "class_9", "class_90", "class_91",
    "class_92", "class_93", "class_94", "class_95", "class_96", "class_97", "class_98",
    "class_99",
]

fine_labels = [
    "Computer vision", "Machine learning", "Distributed computing",
    "Idiopathic Pulmonary Fibrosis", "Irritable Bowel Syndrome", "Kidney Health",
    "Low Testosterone", "Lymphoma", "Medicare", "Menopause", "Migraine",
    "Multiple Sclerosis", "Myelofibrosis", "Structured Storage", "Cancer",
    "Osteoarthritis", "Osteoporosis", "Overactive Bladder", "Parenting",
    "Parkinson's Disease", "Polycythemia Vera", "Psoriasis", "Psoriatic Arthritis",
    "Rheumatoid Arthritis", "Symbolic computation", "Senior Health",
    "Smoking Cessation", "Sports Injuries", "Sprains and Strains", "Stress Management",
    "Molecular biology", "Cell biology", "Human Metabolism", "Immunology", "Genetics",
    "Algorithm design", "Enzymology", "Polymerase chain reaction", "Northern blotting",
    "Southern blotting", "Computer programming", "Data structures", "Bioinformatics",
    "Electricity", "Lorentz force law", "Electrical circuits", "network security",
    "Voltage law", "Digital control", "System identification", "Electrical network",
    "Microcontroller", "Analog signal processing", "Electric motor", "Satellite radio",
    "Control engineering", "Signal-flow graph", "Cryptography",
    "State space representation", "PID controller", "Operational amplifier",
    "Prejudice", "Social cognition", "Person perception", "Nonverbal communication",
    "Prosocial behavior", "Leadership", "Eating disorders", "Operating systems",
    "Depression", "Borderline personality disorder", "Seasonal affective disorder",
    "Schizophrenia", "Antisocial personality disorder", "Media violence",
    "Prenatal development", "Child abuse", "Gender roles", "False memories",
    "Computer graphics", "Attention", "Problem-solving", "computer-aided design",
    "Hydraulics", "Manufacturing engineering", "Machine design", "Fluid mechanics",
    "Internal combustion engine", "Thermodynamics", "Materials Engineering",
    "Image processing", "Strength of materials", "Ambient Intelligence", "Geotextile",
    "Remote Sensing", "Rainwater Harvesting", "Water Pollution", "Suspension Bridge",
    "Stealth Technology", "Green Building", "Solar Energy", "Parallel computing",
    "Construction Management", "Smart Material", "Addiction", "Allergies",
    "Alzheimer's Disease", "Ankylosing Spondylitis", "Anxiety", "Asthma",
    "Atopic Dermatitis", "Atrial Fibrillation", "Relational databases", "Autism",
    "Skin Care", "Bipolar Disorder", "Birth Control", "Children's Health",
    "Crohn's Disease", "Dementia", "Diabetes", "Weight Loss", "Digestive Health",
    "Software engineering", "Emergency Contraception", "Mental Health",
    "Fungal Infection", "Headache", "Healthy Sleep", "Heart Disease", "Hepatitis C",
    "Hereditary Angioedema", "HIV/AIDS", "Hypothyroidism",
]

human_grouping = [
    {
        "Addiction", "Antisocial personality disorder", "Anxiety", "Autism",
        "Bipolar Disorder", "Borderline personality disorder", "Depression",
        "Eating disorders", "Schizophrenia", "Seasonal affective disorder",
    },
    {
        "Gender roles", "Leadership", "Nonverbal communication", "Parenting",
        "Person perception", "Prejudice", "Prosocial behavior", "Social cognition",
    },
    {
        "Attention", "False memories", "Problem-solving",
    },
    {
        "Healthy Sleep", "Skin Care", "Smoking Cessation", "Stress Management",
        "Weight Loss",
    },
    {
        "Children's Health", "Medicare", "Mental Health", "Senior Health",
    },
    {
        "Kidney Health", "Overactive Bladder",
    },
    {
        "Analog signal processing", "Control engineering", "Digital control",
        "Electric motor", "Electrical circuits", "Electrical network", "Electricity",
        "Fluid mechanics", "Hydraulics", "Internal combustion engine",
        "Lorentz force law", "Manufacturing engineering", "Microcontroller",
        "Operational amplifier", "PID controller", "Signal-flow graph",
        "Smart Material", "State space representation", "Strength of materials",
        "System identification", "Thermodynamics", "Voltage law",
        "computer-aided design",
    },
    {
        "Ambient Intelligence", "Geotextile", "Machine design",
        "Materials Engineering", "Rainwater Harvesting", "Remote Sensing",
        "Satellite radio", "Solar Energy", "Stealth Technology", "Suspension Bridge",
        "Water Pollution",
    },
    {
        "Algorithm design", "Bioinformatics", "Computer graphics",
        "Computer programming", "Computer vision", "Cryptography", "Data structures",
        "Distributed computing", "Image processing", "Machine learning",
        "Operating systems", "Parallel computing", "Relational databases",
        "Software engineering", "Structured Storage", "Symbolic computation",
        "network security",
    },
    {
        "Genetics", "Hereditary Angioedema", "Northern blotting",
        "Polymerase chain reaction", "Southern blotting",
    },
    {
        "Alzheimer's Disease", "Dementia", "Headache", "Migraine",
        "Multiple Sclerosis", "Parkinson's Disease",
    },
    {
        "Atopic Dermatitis", "Psoriasis",
    },
    {
        "Diabetes", "Human Metabolism",
    },
    {
        "Cell biology", "Enzymology", "Molecular biology",
    },
    {
        "Allergies", "Immunology", "Lymphoma",
    },
    {
        "Construction Management", "Green Building",
    },
    {
        "Birth Control", "Emergency Contraception", "Prenatal development",
    },
    {
        "Hypothyroidism", "Low Testosterone", "Menopause",
    },
    {
        "Child abuse", "Media violence",
    },
    {
        "Asthma", "Idiopathic Pulmonary Fibrosis",
    },
    {
        "Fungal Infection", "HIV/AIDS", "Hepatitis C",
    },
    {
        "Ankylosing Spondylitis", "Myelofibrosis", "Osteoarthritis", "Osteoporosis",
        "Psoriatic Arthritis", "Rheumatoid Arthritis", "Sports Injuries",
        "Sprains and Strains",
    },
    {
        "Cancer", "Polycythemia Vera",
    },
    {
        "Crohn's Disease", "Digestive Health", "Irritable Bowel Syndrome",
    },
    {
        "Atrial Fibrillation", "Heart Disease",
    },
]

# The same grouping in the {group: [classes]} form used for assortativity.
mappings = {
    "human": {f"group_{i}": sorted(group) for i, group in enumerate(human_grouping, start=1)},
}


def _validate():
    # Every class appears in exactly one group, and codes follow the
    # confusion-matrix order.
    if class_codes != sorted(class_codes) or len(set(fine_labels)) != len(fine_labels):
        raise AssertionError("class codes must be sorted and class names unique")
    flat = [name for group in human_grouping for name in group]
    if sorted(flat) != sorted(fine_labels):
        raise AssertionError("human_grouping must contain every class exactly once")


_validate()
