Queries
-----------
1. Write a query which counts the amount of matches which took place in 2018 and had at least one premium user participating. 
```
SELECT COUNT(*)
FROM user_to_matches
WHERE extract(year from created_at) = 2018 AND membership = 'premium';
```

2. Write a query which finds the list of all users who had at least one winning streak of 3 matches on the platform. A streak here is defined as achieving three or more consecutive wins in the same game. If a user won a match then his/her faction will be the same as the faction in the winner column. 
```
WITH count_streak AS (
  SELECT
    user_id,
    game,
    created_at,
    membership,
    faction,
    winner,
    count(*) OVER (PARTITION BY user_id, game ORDER BY created_at) AS streak
  FROM user_to_matches
  WHERE winner = faction
)
SELECT DISTINCT user_id
FROM count_streak
WHERE streak >= 3;
```