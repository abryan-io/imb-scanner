"""
USPS Service Type Identifier (STID) Lookup Table
Source: USPS STID Table, effective January 21, 2024
https://postalpro.usps.com/mailing/service-type-identifiers

Each entry:
  mail_class          : e.g. "First-Class Mail"
  mail_subclass       : e.g. "Political Mail" / "Ballot Mail" / None
  service_level       : "Basic/Nonautomation" | "Full-Service"
  iv_mtr              : True = with IV MTR, False = without IV MTR
  acs_type            : "OneCode ACS" | "Full-Service ACS" | "Traditional ACS" | None
  address_correction  : human-readable correction option
  flags               : list of applicable footnote symbols ["$","**","***","♻"]
"""

FOOTNOTES = {
    "$":   "Postage and fees are charged for forwarded or returned mail.",
    "*":   "Informed Visibility® Mail Tracking & Reporting (IV® MTR) — "
           "see https://postalpro.usps.com/InformedVisibility",
    "**":  "Requires the printed text ancillary service endorsement. "
           "The option selected must NOT be printed with the endorsement.",
    "***": "CSR Option 2 STIDs may be used for letters and flats but are NOT valid for parcels. "
           "Mailers must establish an ACS account prior to mailing — contact acs@usps.gov "
           "or call 877-640-0724 (option 1).",
    "♻":   "ACS Green & Secure Option. Additional Green & Secure STIDs available with "
           "Secure Destruction enrollment — see https://postalpro.usps.com/mailing/secure-destruction",
    "2":   "Valid only for Periodical copies mailed with alternative addressing "
           "(Occupant, Or Current Resident) in the address block.",
    "3":   "Full-Service Periodical publishers wishing to prevent PS Form 3579 manual address "
           "correction notices should use a Full-Service STID (038 or 045) in the IMb.",
    "4":   "Share Mail requires submission of production quality mailpieces to USPS for approval "
           "prior to use. IV MTR service is also required. "
           "See https://postalpro.usps.com/mailing/share-mail",
    "UOCAVA": "Uniformed and Overseas Citizens Absentee Voting Act voters use only.",
}


def _e(mail_class, service_level, iv_mtr, acs_type, address_correction,
        flags=None, mail_subclass=None):
    """Shorthand entry builder."""
    return {
        "mail_class":         mail_class,
        "mail_subclass":      mail_subclass,
        "service_level":      service_level,
        "iv_mtr":             iv_mtr,
        "acs_type":           acs_type,
        "address_correction": address_correction,
        "flags":              flags or [],
    }


FCM  = "First-Class Mail"
MM   = "USPS Marketing Mail"
PER  = "Periodicals"
BPM  = "Bound Printed Matter"
POL  = "Political Mail"
BAL  = "Ballot Mail"
PRI  = "Priority Mail"
MISC = "Miscellaneous"

BAS  = "Basic/Nonautomation"
FS   = "Full-Service"

OC   = "OneCode ACS"
FSA  = "Full-Service ACS"
TRA  = "Traditional ACS"

Y    = True   # with IV MTR
N    = False  # without IV MTR

STID_TABLE = {
    # ── First-Class Mail ──────────────────────────────────────────────────────
    "300": _e(FCM, BAS, N, None,  "No Address Corrections – No Printed Endorsements"),
    "310": _e(FCM, BAS, Y, None,  "No Address Corrections – No Printed Endorsements"),
    "260": _e(FCM, FS,  N, None,  "No Address Corrections – No Printed Endorsements"),
    "270": _e(FCM, FS,  Y, None,  "No Address Corrections – No Printed Endorsements"),
    "700": _e(FCM, BAS, N, None,  "Manual Corrections", ["**"]),
    "040": _e(FCM, BAS, Y, None,  "Manual Corrections", ["**"]),
    # OneCode ACS
    "230": _e(FCM, BAS, N, OC,   "Address Service Requested Opt 1"),
    "220": _e(FCM, BAS, Y, OC,   "Address Service Requested Opt 1"),
    "080": _e(FCM, BAS, N, OC,   "Address Service Requested Opt 2"),
    "140": _e(FCM, BAS, Y, OC,   "Address Service Requested Opt 2"),
    "504": _e(FCM, BAS, N, OC,   "Change Service Requested Opt 1", ["♻"]),
    "502": _e(FCM, BAS, Y, OC,   "Change Service Requested Opt 1", ["♻"]),
    "082": _e(FCM, BAS, N, OC,   "Change Service Requested Opt 2", ["♻"]),
    "240": _e(FCM, BAS, Y, OC,   "Change Service Requested Opt 2", ["♻"]),
    "341": _e(FCM, BAS, N, OC,   "Return Service Requested Opt 2"),
    "340": _e(FCM, BAS, Y, OC,   "Return Service Requested Opt 2"),
    "345": _e(FCM, BAS, N, OC,   "Temp-Return Service Requested Opt 2"),
    "344": _e(FCM, BAS, Y, OC,   "Temp-Return Service Requested Opt 2"),
    # Full-Service ACS
    "320": _e(FCM, FS,  N, FSA,  "Address Service Requested Opt 1"),
    "314": _e(FCM, FS,  Y, FSA,  "Address Service Requested Opt 1"),
    "081": _e(FCM, FS,  N, FSA,  "Address Service Requested Opt 2"),
    "141": _e(FCM, FS,  Y, FSA,  "Address Service Requested Opt 2"),
    "516": _e(FCM, FS,  N, FSA,  "Change Service Requested Opt 1", ["♻"]),
    "514": _e(FCM, FS,  Y, FSA,  "Change Service Requested Opt 1", ["♻"]),
    "083": _e(FCM, FS,  N, FSA,  "Change Service Requested Opt 2", ["♻"]),
    "241": _e(FCM, FS,  Y, FSA,  "Change Service Requested Opt 2", ["♻"]),
    "343": _e(FCM, FS,  N, FSA,  "Return Service Requested Opt 2"),
    "342": _e(FCM, FS,  Y, FSA,  "Return Service Requested Opt 2"),
    "232": _e(FCM, FS,  N, FSA,  "Temp-Return Service Requested Opt 2"),
    "222": _e(FCM, FS,  Y, FSA,  "Temp-Return Service Requested Opt 2"),
    # Traditional ACS
    "501": _e(FCM, BAS, N, TRA,  "Address Service Requested Opt 1", ["**"]),
    "500": _e(FCM, BAS, Y, TRA,  "Address Service Requested Opt 1", ["**"]),
    "505": _e(FCM, FS,  N, TRA,  "Address Service Requested Opt 1", ["**"]),
    "503": _e(FCM, FS,  Y, TRA,  "Address Service Requested Opt 1", ["**"]),
    "507": _e(FCM, BAS, N, TRA,  "Address Service Requested Opt 2", ["**"]),
    "506": _e(FCM, BAS, Y, TRA,  "Address Service Requested Opt 2", ["**"]),
    "509": _e(FCM, FS,  N, TRA,  "Address Service Requested Opt 2", ["**"]),
    "508": _e(FCM, FS,  Y, TRA,  "Address Service Requested Opt 2", ["**"]),
    "517": _e(FCM, BAS, N, TRA,  "Change Service Requested Opt 1", ["**", "♻"]),
    "515": _e(FCM, BAS, Y, TRA,  "Change Service Requested Opt 1", ["**", "♻"]),
    "521": _e(FCM, FS,  N, TRA,  "Change Service Requested Opt 1", ["**", "♻"]),
    "519": _e(FCM, FS,  Y, TRA,  "Change Service Requested Opt 1", ["**", "♻"]),
    "510": _e(FCM, BAS, N, TRA,  "Change Service Requested Opt 2", ["**", "♻"]),
    "530": _e(FCM, BAS, Y, TRA,  "Change Service Requested Opt 2", ["**", "♻"]),
    "512": _e(FCM, FS,  N, TRA,  "Change Service Requested Opt 2", ["**", "♻"]),
    "511": _e(FCM, FS,  Y, TRA,  "Change Service Requested Opt 2", ["**", "♻"]),
    "535": _e(FCM, BAS, N, TRA,  "Return Service Requested Opt 2", ["**"]),
    "534": _e(FCM, BAS, Y, TRA,  "Return Service Requested Opt 2", ["**"]),
    "537": _e(FCM, FS,  N, TRA,  "Return Service Requested Opt 2", ["**"]),
    "536": _e(FCM, FS,  Y, TRA,  "Return Service Requested Opt 2", ["**"]),
    "543": _e(FCM, BAS, N, TRA,  "Temp-Return Service Requested Opt 2", ["**"]),
    "538": _e(FCM, BAS, Y, TRA,  "Temp-Return Service Requested Opt 2", ["**"]),
    "545": _e(FCM, FS,  N, TRA,  "Temp-Return Service Requested Opt 2", ["**"]),
    "544": _e(FCM, FS,  Y, TRA,  "Temp-Return Service Requested Opt 2", ["**"]),

    # ── Periodicals ───────────────────────────────────────────────────────────
    "704": _e(PER, BAS, N, None, "Manual Corrections"),
    "044": _e(PER, BAS, Y, None, "Manual Corrections"),
    "149": _e(PER, BAS, N, None, "Alternative Address – No Corrections", ["2"]),
    "148": _e(PER, BAS, Y, None, "Alternative Address – No Corrections", ["2"]),
    "147": _e(PER, FS,  N, None, "Alternative Address – No Corrections", ["2"]),
    "146": _e(PER, FS,  Y, None, "Alternative Address – No Corrections", ["2"]),
    "784": _e(PER, BAS, N, OC,  "Address Service Requested"),
    "244": _e(PER, BAS, Y, OC,  "Address Service Requested"),
    "038": _e(PER, FS,  N, FSA, "Address Service Requested", ["3"]),
    "045": _e(PER, FS,  Y, FSA, "Address Service Requested", ["3"]),
    "600": _e(PER, BAS, N, TRA, "Address Service Requested"),
    "599": _e(PER, BAS, Y, TRA, "Address Service Requested"),
    "602": _e(PER, FS,  N, TRA, "Address Service Requested"),
    "601": _e(PER, FS,  Y, TRA, "Address Service Requested"),

    # ── USPS Marketing Mail ───────────────────────────────────────────────────
    "301": _e(MM,  BAS, N, None, "No Address Corrections – No Printed Endorsement"),
    "311": _e(MM,  BAS, Y, None, "No Address Corrections – No Printed Endorsement"),
    "261": _e(MM,  FS,  N, None, "No Address Corrections – No Printed Endorsement"),
    "271": _e(MM,  FS,  Y, None, "No Address Corrections – No Printed Endorsement"),
    "702": _e(MM,  BAS, N, None, "Manual Corrections", ["**"]),
    "042": _e(MM,  BAS, Y, None, "Manual Corrections", ["**"]),
    # OneCode ACS
    "090": _e(MM,  BAS, N, OC,  "Address Service Requested Opt 1", ["**", "$"]),
    "142": _e(MM,  BAS, Y, OC,  "Address Service Requested Opt 1", ["**", "$"]),
    "334": _e(MM,  BAS, N, OC,  "Address Service Requested Opt 2", ["**", "$"]),
    "585": _e(MM,  BAS, Y, OC,  "Address Service Requested Opt 2", ["**", "$"]),
    "092": _e(MM,  BAS, N, OC,  "Change Service Requested Opt 1", ["**", "♻"]),
    "242": _e(MM,  BAS, Y, OC,  "Change Service Requested Opt 1", ["**", "♻"]),
    "513": _e(MM,  BAS, N, OC,  "Change Service Requested Opt 2", ["**", "♻", "$", "***"]),
    "586": _e(MM,  BAS, Y, OC,  "Change Service Requested Opt 2", ["**", "♻", "$", "***"]),
    "272": _e(MM,  BAS, N, OC,  "Return Service Requested Opt 2", ["**", "$"]),
    "262": _e(MM,  BAS, Y, OC,  "Return Service Requested Opt 2", ["**", "$"]),
    # Full-Service ACS
    "091": _e(MM,  FS,  N, FSA, "Address Service Requested Opt 1", ["**", "$"]),
    "143": _e(MM,  FS,  Y, FSA, "Address Service Requested Opt 1", ["**", "$"]),
    "550": _e(MM,  FS,  N, FSA, "Address Service Requested Opt 2", ["**", "$"]),
    "548": _e(MM,  FS,  Y, FSA, "Address Service Requested Opt 2", ["**", "$"]),
    "093": _e(MM,  FS,  N, FSA, "Change Service Requested Opt 1", ["**", "♻"]),
    "243": _e(MM,  FS,  Y, FSA, "Change Service Requested Opt 1", ["**", "♻"]),
    "567": _e(MM,  FS,  N, FSA, "Change Service Requested Opt 2", ["**", "♻", "$", "***"]),
    "231": _e(MM,  FS,  Y, FSA, "Change Service Requested Opt 2", ["**", "♻", "$", "***"]),
    "529": _e(MM,  FS,  N, FSA, "Return Service Requested Opt 2", ["**", "$"]),
    "587": _e(MM,  FS,  Y, FSA, "Return Service Requested Opt 2", ["**", "$"]),
    # Traditional ACS
    "540": _e(MM,  BAS, N, TRA, "Address Service Requested Opt 1", ["**", "$"]),
    "539": _e(MM,  BAS, Y, TRA, "Address Service Requested Opt 1", ["**", "$"]),
    "542": _e(MM,  FS,  N, TRA, "Address Service Requested Opt 1", ["**", "$"]),
    "541": _e(MM,  FS,  Y, TRA, "Address Service Requested Opt 1", ["**", "$"]),
    "547": _e(MM,  BAS, N, TRA, "Address Service Requested Opt 2", ["**", "$"]),
    "546": _e(MM,  BAS, Y, TRA, "Address Service Requested Opt 2", ["**", "$"]),
    "551": _e(MM,  FS,  N, TRA, "Address Service Requested Opt 2", ["**", "$"]),
    "549": _e(MM,  FS,  Y, TRA, "Address Service Requested Opt 2", ["**", "$"]),
    "560": _e(MM,  BAS, N, TRA, "Change Service Requested Opt 1", ["**", "♻"]),
    "559": _e(MM,  BAS, Y, TRA, "Change Service Requested Opt 1", ["**", "♻"]),
    "562": _e(MM,  FS,  N, TRA, "Change Service Requested Opt 1", ["**", "♻"]),
    "561": _e(MM,  FS,  Y, TRA, "Change Service Requested Opt 1", ["**", "♻"]),
    "565": _e(MM,  BAS, N, TRA, "Change Service Requested Opt 2", ["**", "♻", "$", "***"]),
    "564": _e(MM,  BAS, Y, TRA, "Change Service Requested Opt 2", ["**", "♻", "$", "***"]),
    "568": _e(MM,  FS,  N, TRA, "Change Service Requested Opt 2", ["**", "♻", "$", "***"]),
    "566": _e(MM,  FS,  Y, TRA, "Change Service Requested Opt 2", ["**", "♻", "$", "***"]),
    "570": _e(MM,  BAS, N, TRA, "Return Service Requested Opt 2", ["**", "$"]),
    "569": _e(MM,  BAS, Y, TRA, "Return Service Requested Opt 2", ["**", "$"]),
    "572": _e(MM,  FS,  N, TRA, "Return Service Requested Opt 2", ["**", "$"]),
    "571": _e(MM,  FS,  Y, TRA, "Return Service Requested Opt 2", ["**", "$"]),

    # ── Bound Printed Matter ──────────────────────────────────────────────────
    "401": _e(BPM, BAS, N, None, "No Address Corrections – No Printed Endorsement"),
    "451": _e(BPM, BAS, Y, None, "No Address Corrections – No Printed Endorsement"),
    "265": _e(BPM, FS,  N, None, "No Address Corrections – No Printed Endorsement"),
    "351": _e(BPM, FS,  Y, None, "No Address Corrections – No Printed Endorsement"),
    "706": _e(BPM, BAS, N, None, "Manual Corrections", ["**"]),
    "452": _e(BPM, BAS, Y, None, "Manual Corrections", ["**"]),
    # OneCode ACS
    "424": _e(BPM, BAS, N, OC,  "Address Service Requested Opt 1", ["**", "$"]),
    "453": _e(BPM, BAS, Y, OC,  "Address Service Requested Opt 1", ["**", "$"]),
    "605": _e(BPM, BAS, N, OC,  "Address Service Requested Opt 2", ["**", "$"]),
    "454": _e(BPM, BAS, Y, OC,  "Address Service Requested Opt 2", ["**", "$"]),
    "431": _e(BPM, BAS, N, OC,  "Change Service Requested Opt 1", ["**"]),
    "455": _e(BPM, BAS, Y, OC,  "Change Service Requested Opt 1", ["**"]),
    "615": _e(BPM, BAS, N, OC,  "Change Service Requested Opt 2", ["**", "$", "***"]),
    "456": _e(BPM, BAS, Y, OC,  "Change Service Requested Opt 2", ["**", "$", "***"]),
    "619": _e(BPM, BAS, N, OC,  "Return Service Requested Opt 2", ["**", "$"]),
    "457": _e(BPM, BAS, Y, OC,  "Return Service Requested Opt 2", ["**", "$"]),
    # Full-Service ACS
    "423": _e(BPM, FS,  N, FSA, "Address Service Requested Opt 1", ["**", "$"]),
    "353": _e(BPM, FS,  Y, FSA, "Address Service Requested Opt 1", ["**", "$"]),
    "607": _e(BPM, FS,  N, FSA, "Address Service Requested Opt 2", ["**", "$"]),
    "354": _e(BPM, FS,  Y, FSA, "Address Service Requested Opt 2", ["**", "$"]),
    "430": _e(BPM, FS,  N, FSA, "Change Service Requested Opt 1", ["**"]),
    "355": _e(BPM, FS,  Y, FSA, "Change Service Requested Opt 1", ["**"]),
    "617": _e(BPM, FS,  N, FSA, "Change Service Requested Opt 2", ["**", "$", "***"]),
    "356": _e(BPM, FS,  Y, FSA, "Change Service Requested Opt 2", ["**", "$", "***"]),
    "621": _e(BPM, FS,  N, FSA, "Return Service Requested Opt 2", ["**", "$"]),
    "357": _e(BPM, FS,  Y, FSA, "Return Service Requested Opt 2", ["**", "$"]),
    # Traditional ACS
    "603": _e(BPM, BAS, N, TRA, "Address Service Requested Opt 1", ["**", "$"]),
    "458": _e(BPM, BAS, Y, TRA, "Address Service Requested Opt 1", ["**", "$"]),
    "604": _e(BPM, FS,  N, TRA, "Address Service Requested Opt 1", ["**", "$"]),
    "358": _e(BPM, FS,  Y, TRA, "Address Service Requested Opt 1", ["**", "$"]),
    "606": _e(BPM, BAS, N, TRA, "Address Service Requested Opt 2", ["**", "$"]),
    "459": _e(BPM, BAS, Y, TRA, "Address Service Requested Opt 2", ["**", "$"]),
    "608": _e(BPM, FS,  N, TRA, "Address Service Requested Opt 2", ["**", "$"]),
    "359": _e(BPM, FS,  Y, TRA, "Address Service Requested Opt 2", ["**", "$"]),
    "613": _e(BPM, BAS, N, TRA, "Change Service Requested Opt 1", ["**"]),
    "460": _e(BPM, BAS, Y, TRA, "Change Service Requested Opt 1", ["**"]),
    "614": _e(BPM, FS,  N, TRA, "Change Service Requested Opt 1", ["**"]),
    "360": _e(BPM, FS,  Y, TRA, "Change Service Requested Opt 1", ["**"]),
    "616": _e(BPM, BAS, N, TRA, "Change Service Requested Opt 2", ["**", "$", "***"]),
    "461": _e(BPM, BAS, Y, TRA, "Change Service Requested Opt 2", ["**", "$", "***"]),
    "618": _e(BPM, FS,  N, TRA, "Change Service Requested Opt 2", ["**", "$", "***"]),
    "361": _e(BPM, FS,  Y, TRA, "Change Service Requested Opt 2", ["**", "$", "***"]),
    "620": _e(BPM, BAS, N, TRA, "Return Service Requested Opt 2", ["**", "$"]),
    "462": _e(BPM, BAS, Y, TRA, "Return Service Requested Opt 2", ["**", "$"]),
    "622": _e(BPM, FS,  N, TRA, "Return Service Requested Opt 2", ["**", "$"]),
    "362": _e(BPM, FS,  Y, TRA, "Return Service Requested Opt 2", ["**", "$"]),

    # ── Political Mail – First-Class ──────────────────────────────────────────
    "751": _e(FCM, BAS, N, None, "No Address Corrections", mail_subclass=POL),
    "727": _e(FCM, BAS, Y, None, "No Address Corrections", mail_subclass=POL),
    "761": _e(FCM, FS,  N, None, "No Address Corrections", mail_subclass=POL),
    "747": _e(FCM, FS,  Y, None, "No Address Corrections", mail_subclass=POL),
    "752": _e(FCM, BAS, N, None, "Manual Address Corrections", ["**"], mail_subclass=POL),
    "756": _e(FCM, BAS, Y, None, "Manual Address Corrections", ["**"], mail_subclass=POL),
    "753": _e(FCM, BAS, N, OC,  "Address Service Requested Opt 1", mail_subclass=POL),
    "757": _e(FCM, BAS, Y, OC,  "Address Service Requested Opt 1", mail_subclass=POL),
    "754": _e(FCM, BAS, N, OC,  "Address Service Requested Opt 2", mail_subclass=POL),
    "758": _e(FCM, BAS, Y, OC,  "Address Service Requested Opt 2", mail_subclass=POL),
    "755": _e(FCM, BAS, N, OC,  "Change Service Requested Opt 1", ["♻"], mail_subclass=POL),
    "759": _e(FCM, BAS, Y, OC,  "Change Service Requested Opt 1", ["♻"], mail_subclass=POL),
    "763": _e(FCM, FS,  N, FSA, "Address Service Requested Opt 1", mail_subclass=POL),
    "767": _e(FCM, FS,  Y, FSA, "Address Service Requested Opt 1", mail_subclass=POL),
    "764": _e(FCM, FS,  N, FSA, "Address Service Requested Opt 2", mail_subclass=POL),
    "768": _e(FCM, FS,  Y, FSA, "Address Service Requested Opt 2", mail_subclass=POL),
    "765": _e(FCM, FS,  N, FSA, "Change Service Requested Opt 1", ["♻"], mail_subclass=POL),
    "769": _e(FCM, FS,  Y, FSA, "Change Service Requested Opt 1", ["♻"], mail_subclass=POL),
    "762": _e(FCM, FS,  N, FSA, "Return Service Requested Opt 2", mail_subclass=POL),
    "766": _e(FCM, FS,  Y, FSA, "Return Service Requested Opt 2", mail_subclass=POL),

    # ── Political Mail – USPS Marketing Mail ──────────────────────────────────
    "771": _e(MM,  BAS, N, None, "No Address Corrections", mail_subclass=POL),
    "728": _e(MM,  BAS, Y, None, "No Address Corrections", mail_subclass=POL),
    "773": _e(MM,  FS,  N, None, "No Address Corrections", mail_subclass=POL),
    "748": _e(MM,  FS,  Y, None, "No Address Corrections", mail_subclass=POL),
    "772": _e(MM,  BAS, N, None, "Manual Address Corrections", ["**"], mail_subclass=POL),
    "776": _e(MM,  BAS, Y, None, "Manual Address Corrections", ["**"], mail_subclass=POL),
    "775": _e(MM,  BAS, N, OC,  "Change Service Requested Opt 1", ["♻"], mail_subclass=POL),
    "770": _e(MM,  BAS, Y, OC,  "Change Service Requested Opt 1", ["♻"], mail_subclass=POL),
    "774": _e(MM,  FS,  N, FSA, "Change Service Requested Opt 1", ["**", "♻"], mail_subclass=POL),
    "781": _e(MM,  FS,  Y, FSA, "Change Service Requested Opt 1", ["**", "♻"], mail_subclass=POL),
    "785": _e(MM,  BAS, N, TRA, "Change Service Requested Opt 1", ["**", "♻"], mail_subclass=POL),
    "786": _e(MM,  BAS, Y, TRA, "Change Service Requested Opt 1", ["**", "♻"], mail_subclass=POL),

    # ── Ballot Mail – Election Officials to Voter (Outbound) – FCM ───────────
    "715": _e(FCM, BAS, Y, None, "No Address Corrections – No Printed Endorsement", mail_subclass=BAL),
    "720": _e(FCM, FS,  Y, None, "No Address Corrections – No Printed Endorsement", mail_subclass=BAL),
    "716": _e(FCM, BAS, Y, None, "Manual Address Corrections", ["**"], mail_subclass=BAL),
    "717": _e(FCM, BAS, Y, OC,  "Forward Ballot – Address Service Requested Opt 1", mail_subclass=BAL),
    "718": _e(FCM, BAS, Y, OC,  "Forward Ballot – Address Service Requested Opt 2", mail_subclass=BAL),
    "713": _e(FCM, BAS, Y, OC,  "Return Ballot – Return Service Requested Opt 2", mail_subclass=BAL),
    "722": _e(FCM, FS,  Y, FSA, "Forward Ballot – Address Service Requested Opt 1", mail_subclass=BAL),
    "723": _e(FCM, FS,  Y, FSA, "Forward Ballot – Address Service Requested Opt 2", mail_subclass=BAL),
    "725": _e(FCM, FS,  Y, FSA, "Return Ballot – Return Service Requested Opt 2", mail_subclass=BAL),

    # ── Ballot Mail – Election Officials to Voter (Outbound) – USPS MM ───────
    "735": _e(MM,  BAS, Y, None, "No Address Corrections – No Printed Endorsement", mail_subclass=BAL),
    "741": _e(MM,  FS,  Y, None, "No Address Corrections – No Printed Endorsement", mail_subclass=BAL),
    "736": _e(MM,  BAS, Y, None, "Manual Address Corrections", ["**"], mail_subclass=BAL),
    "737": _e(MM,  BAS, Y, OC,  "Forward Ballot – Address Service Requested Opt 1", ["**", "$"], mail_subclass=BAL),
    "738": _e(MM,  BAS, Y, OC,  "Forward Ballot – Address Service Requested Opt 2", ["**", "$"], mail_subclass=BAL),
    "714": _e(MM,  BAS, Y, OC,  "Return Ballot – Return Service Requested Opt 2", ["**", "$"], mail_subclass=BAL),
    "743": _e(MM,  FS,  Y, FSA, "Forward Ballot – Address Service Requested Opt 1", ["**", "$"], mail_subclass=BAL),
    "744": _e(MM,  FS,  Y, FSA, "Forward Ballot – Address Service Requested Opt 2", ["**", "$"], mail_subclass=BAL),
    "746": _e(MM,  FS,  Y, FSA, "Return Ballot – Return Service Requested Opt 2", ["**", "$"], mail_subclass=BAL),
    "726": _e(MM,  BAS, Y, TRA, "Return Ballot – Return Service Requested Opt 2", ["**", "$"], mail_subclass=BAL),

    # ── Ballot Mail – Voter to Election Officials (Return/Inbound) ────────────
    "777": _e(FCM, BAS, N, None, "First-Class Mail Reply", mail_subclass=BAL),
    "778": _e(FCM, BAS, N, None, "Business Reply Mail", mail_subclass=BAL),
    "779": _e(FCM, BAS, N, None, "Permit Reply Mail", mail_subclass=BAL),
    "780": _e(FCM, BAS, N, None, "UOCAVA", ["UOCAVA"], mail_subclass=BAL),

    # ── Priority Mail ─────────────────────────────────────────────────────────
    "710": _e(PRI, BAS, N, None, "Priority Mail"),
    "712": _e(PRI, BAS, N, None, "Priority Mail Flat Rate"),

    # ── Miscellaneous / Reply Mail ────────────────────────────────────────────
    "703": _e(MISC, BAS, N, None, "Courtesy Reply Mail (by ZIP)"),
    "050": _e(MISC, BAS, Y, None, "Courtesy Reply Mail (by ZIP)"),
    "708": _e(MISC, BAS, N, None, "Business Reply Mail (by ZIP)"),
    "052": _e(MISC, BAS, Y, None, "Business Reply Mail (by ZIP)"),
    "701": _e(MISC, BAS, N, None, "First-Class Reply Mail / PRM (by ZIP)"),
    "051": _e(MISC, BAS, Y, None, "First-Class Reply Mail / PRM (by ZIP)"),
    "070": _e(MISC, BAS, N, None, "Courtesy Reply Mail (by MID)"),
    "030": _e(MISC, BAS, Y, None, "Courtesy Reply Mail (by MID)"),
    "072": _e(MISC, BAS, N, None, "Business Reply Mail (by MID)"),
    "032": _e(MISC, BAS, Y, None, "Business Reply Mail (by MID)"),
    "071": _e(MISC, BAS, N, None, "First-Class Reply Mail (by MID)"),
    "031": _e(MISC, BAS, Y, None, "First-Class Reply Mail (by MID)"),
    "733": _e(MISC, BAS, Y, None, "Share Mail with a Unique IMb", ["4"]),
    "734": _e(MISC, BAS, Y, None, "Share Mail with a Static IMb", ["4"]),
}


def lookup(stid: str) -> dict | None:
    """
    Look up a STID string. Returns the entry dict with an added 'note_texts' list,
    or None if not found.
    """
    key = str(stid).strip().zfill(3)
    entry = STID_TABLE.get(key)
    if entry is None:
        return None
    result = dict(entry)
    result["stid"] = key
    result["note_texts"] = [
        FOOTNOTES[f] for f in entry["flags"] if f in FOOTNOTES
    ]
    return result


def describe(stid: str) -> str:
    """Return a concise one-line description for a STID."""
    e = lookup(stid)
    if e is None:
        return f"Unknown STID {stid}"
    parts = [e["mail_class"]]
    if e["mail_subclass"]:
        parts[0] += f" ({e['mail_subclass']})"
    if e["acs_type"]:
        parts.append(e["acs_type"])
    parts.append(e["address_correction"])
    parts.append(e["service_level"])
    parts.append("with IV MTR" if e["iv_mtr"] else "w/o IV MTR")
    if e["flags"]:
        parts.append(" ".join(e["flags"]))
    return " · ".join(parts)


if __name__ == "__main__":
    # Quick test
    for stid in ["243", "300", "310", "042", "784", "710", "715", "777", "999"]:
        print(f"{stid}: {describe(stid)}")
