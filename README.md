+----------------------------------------------------+
|                    HDFS INPUT                      |
|----------------------------------------------------|
|  /user/sandeep/input/email_interactions.csv        |
|  (100,001 records)                                 |
+--------------------------+-------------------------+
                           |
                           v
+----------------------------------------------------+
|                    MAP PHASE                       |
|----------------------------------------------------|
| Mapper Logic:                                      |
|  - Read each email interaction record              |
|  - Extract country, product_id, event_type         |
|                                                    |
| Mapper Output (Key → Value):                        |
|  (country | product_id | event_type) → 1           |
|                                                    |
| Example:                                           |
|  (India | PROD005 | OPENED) → 1                     |
+--------------------------+-------------------------+
                           |
                           v
+----------------------------------------------------+
|              SHUFFLE & SORT PHASE                  |
|----------------------------------------------------|
| Hadoop groups mapper output by key:                |
|                                                    |
|  (India | PROD005 | OPENED) → [1,1,1,...]           |
|  (India | PROD005 | SENT)   → [1,1,1,...]           |
|  (USA   | PROD010 | BOUNCED)→ [1,1,1,...]           |
+--------------------------+-------------------------+
                           |
                           v
+----------------------------------------------------+
|                   REDUCE PHASE                    |
|----------------------------------------------------|
| Reducer Logic:                                     |
|  - Receive grouped values per key                  |
|  - Sum all values                                  |
|                                                    |
| Reducer Output:                                    |
|  (country | product_id | event_type) → total_count |
|                                                    |
| Example:                                           |
|  (India | PROD005 | OPENED) → 1240                  |
+--------------------------+-------------------------+
                           |
                           v
+----------------------------------------------------+
|                   HDFS OUTPUT                     |
|----------------------------------------------------|
|  /user/sandeep/output/analysis1_email/             |
|                                                    |
|  Output Records: 480                                |
|                                                    |
|  Sample Output:                                    |
|  India|PROD005|OPENED   1240                        |
|  India|PROD005|SENT     3980                        |
|  India|PROD005|BOUNCED  210                         |
+----------------------------------------------------+
