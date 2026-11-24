import pandas as pd

from scripts.reserve_markets import load_reserve_inputs


def test_load_reserve_inputs_corrects_26_april_2026_mfrr_values(tmp_path):
    hours = pd.date_range("2026-04-25 22:00", "2026-04-26 07:00", freq="h")
    reserve_folder = tmp_path / "reserve_markets"
    reserve_folder.mkdir()
    columns = [
        "mFRRSAUpReqMW",
        "mFRRSADownReqMW",
        "TotalmFRRUpMW",
        "TotalmFRRDownMW",
    ]
    activation = pd.DataFrame(
        100.0,
        index=pd.date_range(hours[0], hours[-1], freq="15min"),
        columns=columns,
    )
    activation.to_csv(reserve_folder / "reserves_mFRR_activation.csv", sep=";")

    for filename in (
        "reserves_aFRR_capacity.csv",
        "reserves_aFRR_activation.csv",
        "reserves_mFRR_capacity.csv",
        "reserves_FCR_DK1.csv",
    ):
        pd.DataFrame({"value": 1.0}, index=hours).to_csv(
            reserve_folder / filename, sep=";"
        )

    reserves = load_reserve_inputs(tmp_path, hours)
    result = reserves["mFRR_energy_activation"]

    assert result.loc[hours[0], columns].eq(100.0).all()
    assert result.loc[hours[1:9], columns].eq(0.0).all().all()
    assert result.loc[hours[9], columns].eq(100.0).all()
    assert set(reserves) == {
        "aFRR_capacity_reserves",
        "aFRR_energy_activation",
        "mFRR_capacity_reserves",
        "mFRR_energy_activation",
        "FCR_capacity_reserves",
        "FCR_activation_15min",
    }
