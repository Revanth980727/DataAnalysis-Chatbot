examples = """
[
    {
        "question": "Give me the Customers for those who have multiple tech orders and truck rolls",
        "sql": "SELECT a.`Account Number`, a.`Customer Name`, a.`Customer Location`, t.`Disconnect Reason` FROM `account` a JOIN `techorder` t ON a.`Account Number` = t.`Account Number` WHERE t.`Truck Roll` = 'Y' GROUP BY a.`Account Number`, a.`Customer Location`, a.`Customer Name`, t.`Disconnect Reason` HAVING COUNT(t.`Tech Order`) > 1;"
    },

    {
        "question": "Who are the customers that got impacted due Outage on June 06 the in the area of Saint Louis.",
        "sql": "SELECT a.`Account Number`, a.`Customer Name`, o.`Outage Start Time`, o.`Outage End Time`, o.`Outage Reason`, o.`1st Informed Date`, o.`2nd Informed Date` FROM `account` a JOIN (SELECT `Account Number`, `Outage Start Time`, `Outage End Time`, `Outage Reason`, `1st Informed Date`, `2nd Informed Date` FROM `outage` WHERE `Date` = '6/6/2024' AND `Customer Location` LIKE '%Saint Louis%') o ON a.`Account Number` = o.`Account Number`;"
    },

    {
        "question": "What are the new connections that are established in June 2024 from each competitor",
        "sql": "SELECT competitor AS `Competitor`,SUM(CASE WHEN `Source` = 'Port IN Cable' THEN `New Connections` ELSE 0 END) AS `Port IN Cable`, SUM(CASE WHEN `Source` = 'Port In Mobile' THEN `New Connections` ELSE 0 END) AS `Port In Mobile`, GROUP_CONCAT(DISTINCT `Account Number` ORDER BY `Account Number` ASC SEPARATOR ', ') AS `Account Numbers`, GROUP_CONCAT(DISTINCT `Customer Name` ORDER BY `Customer Name` ASC SEPARATOR ', ') AS `Customer Names` FROM (SELECT 'Port IN Cable' AS `Source`, a.`Account Number`, a.`Customer Name`, competitor AS `Competitor`, COUNT(*) AS `New Connections` FROM (SELECT DISTINCT `Port IN Cable` AS competitor, `Start Date` FROM `account` WHERE MONTH(STR_TO_DATE(`Start Date`, '%m/%d/%Y')) = 6 AND `Port IN Cable` IS NOT NULL) AS subquery JOIN `account` a ON subquery.competitor = a.`Port IN Cable` AND a.`Start Date` = subquery.`Start Date` GROUP BY `Source`, a.`Account Number`, a.`Customer Name`, competitor UNION ALL SELECT 'Port In Mobile' AS `Source`, a.`Account Number`, a.`Customer Name`, competitor AS `Competitor`, COUNT(*) AS `New Connections` FROM (SELECT DISTINCT `Port In Mobile` AS competitor, `Start Date` FROM `account` WHERE MONTH(STR_TO_DATE(`Start Date`, '%m/%d/%Y')) = 6 AND `Port In Mobile` IS NOT NULL) AS subquery JOIN `account` a ON subquery.competitor = a.`Port In Mobile` AND a.`Start Date` = subquery.`Start Date` GROUP BY `Source`, a.`Account Number`, a.`Customer Name`, competitor) AS combined_results GROUP BY competitor;"
    },
    {
        "question": "Who are the customers that left Charter due to unresolved issues",
        "sql": "SELECT COUNT(DISTINCT t.`Account Number`) AS `Customers Left Charter`, t.`Account Number`, a.`Customer Name` FROM `techorder` t JOIN `account` a ON t.`Account Number` = a.`Account Number` WHERE t.`Resolved/Unresolved` = 'N' AND a.`End Date` <> '1/1/9999' GROUP BY t.`Account Number`, a.`Customer Name`;"
    },
    {
        "question": "Why did they leave Charter?",
        "sql": "SELECT t.`Account Number`, a.`Customer Name`, MAX(t.`Disconnect Reason`) AS `Disconnect Reason` FROM `techorder` t JOIN `account` a ON t.`Account Number` = a.`Account Number` WHERE t.`Resolved/Unresolved` = 'N' AND a.`End Date` <> '1/1/9999' GROUP BY t.`Account Number`, a.`Customer Name`;"
    },
    {
        "question": "How many customers left Charter due to unresolved issues",
        "sql": "SELECT COUNT(DISTINCT t.`Account Number`) AS `Customers Left Charter` FROM `techorder` t JOIN `account` a ON t.`Account Number` = a.`Account Number` WHERE t.`Resolved/Unresolved` = 'N' AND a.`End Date` <> '1/1/9999';"
    },
    {
        "question": "What is the primary reason for a tech order not being resolved",
        "sql" : "SELECT DISTINCT cd.`Call Reason` AS `Call Reason`, t.`Disconnect Reason` AS `Disconnect Reason`, a.`Account Number` AS `Account Number`, a.`Customer Name` AS `Customer Name`, a.`Customer Location` AS `Customer Location` FROM `calldetails` cd JOIN `techorder` t ON cd.`Tech Order Number` = t.`Tech Order` JOIN `account` a ON t.`Account Number` = a.`Account Number` WHERE cd.`Tech Order Flag` = 'Y' AND t.`Resolved/Unresolved` = 'N';"
    },
    {
        "question": "What are the top 3 reasons for calls that resulted in technical orders",
        "sql": "SELECT `Call Reason`, COUNT(*) AS `Count` FROM `calldetails` WHERE `Tech Order Flag` = 'Y' GROUP BY `Call Reason` ORDER BY COUNT(*) DESC LIMIT 3;"
    },
    {
        "question": "Which customers have had both a technical order and an outage in June 2024.",
        "sql": "SELECT DISTINCT f.`Account Number` AS `Account Number`, o.`Outage Reason` AS `Outage Reason`, cd.`Call Reason` AS `Call Reason` FROM `calldetails` cd JOIN `facttable` f ON cd.`CallIDSK` = f.`CallIDSK` JOIN `outage` o ON f.`Account Number` = o.`Account Number` WHERE cd.`Tech Order Flag` = 'Y' AND MONTH(STR_TO_DATE(o.`Date`, '%m/%d/%Y')) = 6 AND YEAR(STR_TO_DATE(o.`Date`, '%m/%d/%Y')) = 2024;"
    },
    {
        "question": "What is the average duration of outages for each customer location, and how does it compare to the overall average outage duration?",
        "sql" : "WITH location_avg_duration AS (SELECT `Customer Location`, AVG(TIMESTAMPDIFF(MINUTE, STR_TO_DATE(`Outage Start Time`, '%h:%i:%s %p'), STR_TO_DATE(`Outage End Time`, '%h:%i:%s %p'))) AS avg_duration FROM outage GROUP BY `Customer Location`), overall_avg_duration AS (SELECT AVG(TIMESTAMPDIFF(MINUTE, STR_TO_DATE(`Outage Start Time`, '%h:%i:%s %p'), STR_TO_DATE(`Outage End Time`, '%h:%i:%s %p'))) AS overall_avg FROM outage) SELECT lad.`Customer Location`, lad.avg_duration, oad.overall_avg, CASE WHEN lad.avg_duration > oad.overall_avg THEN 'Above Average' WHEN lad.avg_duration < oad.overall_avg THEN 'Below Average' ELSE 'Average' END AS comparison FROM location_avg_duration lad CROSS JOIN overall_avg_duration oad ORDER BY lad.avg_duration DESC;"
    },
    {
        "question": "What is the average duration of outages for each customer location, and how does it compare to the overall average outage duration and who are the agents that attend the call",
        "sql": "WITH location_avg_duration AS (SELECT o.`Customer Location`, a.`Agent Name`, AVG(TIMESTAMPDIFF(MINUTE, STR_TO_DATE(o.`Outage Start Time`, '%h:%i:%s %p'), STR_TO_DATE(o.`Outage End Time`, '%h:%i:%s %p'))) AS avg_duration,COUNT(*) AS num_records FROM outage o JOIN facttable f ON o.`Account Number` = f.`Account Number` JOIN agent a ON f.`AgentKey` = a.`AgentKey` GROUP BY o.`Customer Location`, a.`Agent Name`), overall_avg_duration AS (SELECT AVG(TIMESTAMPDIFF(MINUTE, STR_TO_DATE(`Outage Start Time`, '%h:%i:%s %p'), STR_TO_DATE(`Outage End Time`, '%h:%i:%s %p'))) AS overall_avg FROM outage) SELECT lad.`Customer Location`, lad.`Agent Name`, lad.avg_duration, lad.num_records, oad.overall_avg, CASE WHEN lad.avg_duration > oad.overall_avg THEN 'Above Average' WHEN lad.avg_duration < oad.overall_avg THEN 'Below Average' ELSE 'Average' END AS comparison FROM location_avg_duration lad CROSS JOIN overall_avg_duration oad ORDER BY lad.`Customer Location`, lad.avg_duration DESC;"
    },
    {
        "question": "which internet service has the most issues.",
        "sql": "SELECT a.`Internet Service`,a.`Account Number`,a.`Customer Location`, COUNT(*) AS `Number of Issues` FROM calldetails cd JOIN facttable f ON cd.`CallIDSK` = f.`CallIDSK` JOIN account a ON f.`Account Number` = a.`Account Number` WHERE cd.`Call Reason` LIKE '%internet%' GROUP BY a.`Internet Service`, a.`Account Number`, a.`Customer Location` ORDER BY `Number of Issues` DESC;"
    }
]
"""