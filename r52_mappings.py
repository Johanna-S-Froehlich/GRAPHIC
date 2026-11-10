"""Class metadata for R52 (52-class single-label Reuters-21578 subset).

Exposes what ``metrics.py`` looks up:
  - ``class_codes``:    the 52 R52 label codes, in confusion-matrix order (sorted)
  - ``fine_labels``:    the class names, in the same order
  - ``human_grouping``: the human baseline grouping of the classes (9 groups)
  - ``mappings``:       the same grouping in the form used for assortativity
"""

class_codes = [
    "acq", "alum", "bop", "carcass", "cocoa", "coffee", "copper", "cotton", "cpi",
    "cpu", "crude", "dlr", "earn", "fuel", "gas", "gnp", "gold", "grain", "heat",
    "housing", "income", "instal-debt", "interest", "ipi", "iron-steel", "jet", "jobs",
    "lead", "lei", "livestock", "lumber", "meal-feed", "money-fx", "money-supply",
    "nat-gas", "nickel", "orange", "pet-chem", "platinum", "potato", "reserves",
    "retail", "rubber", "ship", "strategic-metal", "sugar", "tea", "tin", "trade",
    "veg-oil", "wpi", "zinc",
]

fine_labels = [
    "Mergers/Acquisitions", "Aluminum", "Balance of Payments", "Carcass", "Cocoa",
    "Coffee", "Copper", "Cotton", "Consumer Price Index", "Capacity Utilization",
    "Crude Oil", "U.S. Dollar", "Earnings and Earnings Forecasts", "Fuel Oil",
    "Gasoline", "Gross National/Domestic Product", "Gold", "Grain",
    "Heating Oil / Gas Oil", "Housing Starts", "Personal Income",
    "Instalment Debt / Consumer Credit", "Interest Rates",
    "Industrial Production Index", "Iron and Steel", "Jet and Kerosene",
    "Unemployment", "Lead", "Leading Economic Indicators", "Livestock", "Lumber",
    "Meal / Feed", "Money/Foreign Exchange", "Money Supply", "Natural Gas", "Nickel",
    "Orange", "Petro-Chemicals", "Platinum", "Potato", "Reserves", "Retail Sales",
    "Rubber", "Shipping", "Strategic Metal", "Sugar", "Tea", "Tin", "Trade",
    "Vegetable Oil", "Wholesale Price Index", "Zinc",
]

human_grouping = [
    {
        "Crude Oil", "Fuel Oil", "Gasoline", "Heating Oil / Gas Oil",
        "Jet and Kerosene", "Natural Gas", "Petro-Chemicals",
    },
    {
        "Consumer Price Index", "Industrial Production Index", "Wholesale Price Index",
    },
    {
        "Earnings and Earnings Forecasts", "Gross National/Domestic Product",
        "Personal Income", "Retail Sales", "Trade",
    },
    {
        "Cotton", "Lumber", "Rubber",
    },
    {
        "Cocoa", "Coffee", "Grain", "Orange", "Potato", "Sugar", "Tea",
        "Vegetable Oil",
    },
    {
        "Carcass", "Livestock", "Meal / Feed",
    },
    {
        "Balance of Payments", "Instalment Debt / Consumer Credit", "Interest Rates",
        "Money Supply", "Money/Foreign Exchange", "Reserves", "U.S. Dollar",
    },
    {
        "Capacity Utilization", "Housing Starts", "Leading Economic Indicators",
        "Mergers/Acquisitions", "Shipping", "Unemployment",
    },
    {
        "Aluminum", "Copper", "Gold", "Iron and Steel", "Lead", "Nickel", "Platinum",
        "Strategic Metal", "Tin", "Zinc",
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
