+------------------------------------------------------+
|               Input Dataset (HDFS)                   |
|                                                      |
|  File: email_interactions.csv                        |
|  Fields:                                             |
|   - email_event_id                                  |
|   - hcp_id                                          |
|   - hco_id                                          |
|   - campaign_id                                     |
|   - product_id                                      |
|   - event_type (SENT / OPENED / BOUNCED)             |
|   - country                                         |
|   - event_timestamp                                 |
+------------------------------------------------------+
                        |
                        v
+------------------------------------------------------+
|                    Map Phase                         |
|                                                      |
|  Mapper Input: One email event record                |
|                                                      |
|  Mapper Logic:                                       |
|   - Extract country                                 |
|   - Extract product_id                              |
|   - Extract event_type                              |
|                                                      |
|  Mapper Output (Key, Value):                         |
|   (country|product_id|event_type , 1)               |
|                                                      |
|  Example:                                           |
|   (India|PROD005|OPENED , 1)                         |
+------------------------------------------------------+
                        |
                        v
+------------------------------------------------------+
|               Shuffle & Sort Phase                   |
|                                                      |
|  Hadoop groups identical keys together               |
|                                                      |
|  Example Groups:                                     |
|   (India|PROD005|OPENED)  -> [1,1,1,...]             |
|   (India|PROD005|SENT)    -> [1,1,1,...]             |
|   (USA|PROD010|BOUNCED)   -> [1,1,...]               |
+------------------------------------------------------+
                        |
                        v
+------------------------------------------------------+
|                   Reduce Phase                       |
|                                                      |
|  Reducer Logic:                                     |
|   - Sum all values for each key                      |
|                                                      |
|  Reducer Output:                                    |
|   (country|product_id|event_type , total_count)     |
|                                                      |
|  Example:                                           |
|   (India|PROD005|OPENED , 1240)                      |
|   (India|PROD005|SENT , 3980)                        |
+------------------------------------------------------+
                        |
                        v
+------------------------------------------------------+
|                Final Output (HDFS)                   |
|                                                      |
|  Path: /user/sandeep/output/analysis1_email          |
|  Output Records: 480                                 |
|                                                      |
|  Sample Output:                                     |
|   India|PROD005|OPENED   1240                        |
|   India|PROD005|SENT     3980                        |
|   India|PROD005|BOUNCED  210                         |
+------------------------------------------------------+
