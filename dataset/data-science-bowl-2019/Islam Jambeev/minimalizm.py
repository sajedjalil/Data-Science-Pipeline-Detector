import pandas as pd

test_df = pd.read_csv('../input/data-science-bowl-2019/test.csv')
installation_id = []
accuracy_group = []

test_users = test_df.groupby(["installation_id"])
for user in test_users:
    game_sessions = user[1].iloc[-1]["game_session"]
    title = user[1].iloc[-1]["title"]
    title = title.split(" (")[0]
    installation_id.append(user[0])
    if title == "Bird Measurer" or title == "Chest Sorter":
        accuracy_group.append(0)
    else:
        accuracy_group.append(3)

submission = pd.DataFrame({
    "installation_id": installation_id,
    "accuracy_group": accuracy_group,
})

submission.to_csv('submission.csv', index=False)