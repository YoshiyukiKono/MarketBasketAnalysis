-- 
-- Script to prepare data for a Market Basket Analysis algorithm
-- Sample data used: Public dataset for Supermarkt Data :
--                   Number of Sales Records            : 3.346.083
--                   Maximum number of lineitems        :        97
--

--
-- Create hive definition for the big Point Of Sale table:
-- Each record is 1 Sales order. The LineItems are concatenated in the line, separated by a '|'
-- Using the hive 'array' construct 
--

drop table transactions_andre_all;
CREATE external table if not exists transactions_andre_all(
    id BIGINT,
    user_id BIGINT,
    order_number BIGINT,
    order_dow INT,
    order_hour_of_day INT,
    days_since_prior_order INT,
    item array<string>
    )
    row format delimited
    fields terminated by ','
    collection items terminated by '|'
    STORED AS TEXTFILE LOCATION '/user/clouderanA/Andre/transactions_big'
    tblproperties ("skip.header.line.count"="1");

--
-- Create an temporary table, with just the ID and teh ITEM fields.
-- But the table will be transposed, meaning each ID/ITEM combination will be a separate record.
-- Also, Data Quality needs to be improved. with regexp_replace, some Quote characters are removed.
--

drop table transactions_andre_temp;
create table transactions_andre_temp as
select id, regexp_replace(item[0], '"', '') as item from transactions_andre_all union all
select id, regexp_replace(item[1], '"', '') as item from transactions_andre_all where item[1] is not null union all
select id, regexp_replace(item[2], '"', '') as item from transactions_andre_all where item[2] is not null union all
select id, regexp_replace(item[3], '"', '') as item from transactions_andre_all where item[3] is not null union all
select id, regexp_replace(item[4], '"', '') as item from transactions_andre_all where item[4] is not null union all
select id, regexp_replace(item[5], '"', '') as item from transactions_andre_all where item[5] is not null union all
select id, regexp_replace(item[6], '"', '') as item from transactions_andre_all where item[6] is not null union all
select id, regexp_replace(item[7], '"', '') as item from transactions_andre_all where item[7] is not null union all
select id, regexp_replace(item[8], '"', '') as item from transactions_andre_all where item[8] is not null union all
select id, regexp_replace(item[9], '"', '') as item from transactions_andre_all where item[9] is not null union all
select id, regexp_replace(item[10], '"', '') as item from transactions_andre_all where item[10] is not null union all
select id, regexp_replace(item[11], '"', '') as item from transactions_andre_all where item[11] is not null union all
select id, regexp_replace(item[12], '"', '') as item from transactions_andre_all where item[12] is not null union all
select id, regexp_replace(item[13], '"', '') as item from transactions_andre_all where item[13] is not null union all
select id, regexp_replace(item[14], '"', '') as item from transactions_andre_all where item[14] is not null union all
select id, regexp_replace(item[15], '"', '') as item from transactions_andre_all where item[15] is not null union all
select id, regexp_replace(item[16], '"', '') as item from transactions_andre_all where item[16] is not null union all
select id, regexp_replace(item[17], '"', '') as item from transactions_andre_all where item[17] is not null union all
select id, regexp_replace(item[18], '"', '') as item from transactions_andre_all where item[18] is not null union all
select id, regexp_replace(item[19], '"', '') as item from transactions_andre_all where item[19] is not null union all
select id, regexp_replace(item[20], '"', '') as item from transactions_andre_all where item[20] is not null union all
select id, regexp_replace(item[21], '"', '') as item from transactions_andre_all where item[21] is not null union all
select id, regexp_replace(item[22], '"', '') as item from transactions_andre_all where item[22] is not null union all
select id, regexp_replace(item[23], '"', '') as item from transactions_andre_all where item[23] is not null union all
select id, regexp_replace(item[24], '"', '') as item from transactions_andre_all where item[24] is not null union all
select id, regexp_replace(item[25], '"', '') as item from transactions_andre_all where item[25] is not null union all
select id, regexp_replace(item[26], '"', '') as item from transactions_andre_all where item[26] is not null union all
select id, regexp_replace(item[27], '"', '') as item from transactions_andre_all where item[27] is not null union all
select id, regexp_replace(item[28], '"', '') as item from transactions_andre_all where item[28] is not null union all
select id, regexp_replace(item[29], '"', '') as item from transactions_andre_all where item[29] is not null union all
select id, regexp_replace(item[30], '"', '') as item from transactions_andre_all where item[30] is not null union all
select id, regexp_replace(item[31], '"', '') as item from transactions_andre_all where item[31] is not null union all
select id, regexp_replace(item[32], '"', '') as item from transactions_andre_all where item[32] is not null union all
select id, regexp_replace(item[33], '"', '') as item from transactions_andre_all where item[33] is not null union all
select id, regexp_replace(item[34], '"', '') as item from transactions_andre_all where item[34] is not null union all
select id, regexp_replace(item[35], '"', '') as item from transactions_andre_all where item[35] is not null union all
select id, regexp_replace(item[36], '"', '') as item from transactions_andre_all where item[36] is not null union all
select id, regexp_replace(item[37], '"', '') as item from transactions_andre_all where item[37] is not null union all
select id, regexp_replace(item[38], '"', '') as item from transactions_andre_all where item[38] is not null union all
select id, regexp_replace(item[39], '"', '') as item from transactions_andre_all where item[39] is not null union all
select id, regexp_replace(item[40], '"', '') as item from transactions_andre_all where item[40] is not null union all
select id, regexp_replace(item[41], '"', '') as item from transactions_andre_all where item[41] is not null union all
select id, regexp_replace(item[42], '"', '') as item from transactions_andre_all where item[42] is not null union all
select id, regexp_replace(item[43], '"', '') as item from transactions_andre_all where item[43] is not null union all
select id, regexp_replace(item[44], '"', '') as item from transactions_andre_all where item[44] is not null union all
select id, regexp_replace(item[45], '"', '') as item from transactions_andre_all where item[45] is not null union all
select id, regexp_replace(item[46], '"', '') as item from transactions_andre_all where item[46] is not null union all
select id, regexp_replace(item[47], '"', '') as item from transactions_andre_all where item[47] is not null union all
select id, regexp_replace(item[48], '"', '') as item from transactions_andre_all where item[48] is not null union all
select id, regexp_replace(item[49], '"', '') as item from transactions_andre_all where item[49] is not null union all
select id, regexp_replace(item[50], '"', '') as item from transactions_andre_all where item[50] is not null union all
select id, regexp_replace(item[51], '"', '') as item from transactions_andre_all where item[51] is not null union all
select id, regexp_replace(item[52], '"', '') as item from transactions_andre_all where item[52] is not null union all
select id, regexp_replace(item[53], '"', '') as item from transactions_andre_all where item[53] is not null union all
select id, regexp_replace(item[54], '"', '') as item from transactions_andre_all where item[54] is not null union all
select id, regexp_replace(item[55], '"', '') as item from transactions_andre_all where item[55] is not null union all
select id, regexp_replace(item[56], '"', '') as item from transactions_andre_all where item[56] is not null union all
select id, regexp_replace(item[57], '"', '') as item from transactions_andre_all where item[57] is not null union all
select id, regexp_replace(item[58], '"', '') as item from transactions_andre_all where item[58] is not null union all
select id, regexp_replace(item[59], '"', '') as item from transactions_andre_all where item[59] is not null union all
select id, regexp_replace(item[60], '"', '') as item from transactions_andre_all where item[60] is not null union all
select id, regexp_replace(item[61], '"', '') as item from transactions_andre_all where item[61] is not null union all
select id, regexp_replace(item[62], '"', '') as item from transactions_andre_all where item[62] is not null union all
select id, regexp_replace(item[63], '"', '') as item from transactions_andre_all where item[63] is not null union all
select id, regexp_replace(item[64], '"', '') as item from transactions_andre_all where item[64] is not null union all
select id, regexp_replace(item[65], '"', '') as item from transactions_andre_all where item[65] is not null union all
select id, regexp_replace(item[66], '"', '') as item from transactions_andre_all where item[66] is not null union all
select id, regexp_replace(item[67], '"', '') as item from transactions_andre_all where item[67] is not null union all
select id, regexp_replace(item[68], '"', '') as item from transactions_andre_all where item[68] is not null union all
select id, regexp_replace(item[69], '"', '') as item from transactions_andre_all where item[69] is not null union all
select id, regexp_replace(item[70], '"', '') as item from transactions_andre_all where item[70] is not null union all
select id, regexp_replace(item[71], '"', '') as item from transactions_andre_all where item[71] is not null union all
select id, regexp_replace(item[72], '"', '') as item from transactions_andre_all where item[72] is not null union all
select id, regexp_replace(item[73], '"', '') as item from transactions_andre_all where item[73] is not null union all
select id, regexp_replace(item[74], '"', '') as item from transactions_andre_all where item[74] is not null union all
select id, regexp_replace(item[75], '"', '') as item from transactions_andre_all where item[75] is not null union all
select id, regexp_replace(item[76], '"', '') as item from transactions_andre_all where item[76] is not null union all
select id, regexp_replace(item[77], '"', '') as item from transactions_andre_all where item[77] is not null union all
select id, regexp_replace(item[78], '"', '') as item from transactions_andre_all where item[78] is not null union all
select id, regexp_replace(item[79], '"', '') as item from transactions_andre_all where item[79] is not null union all
select id, regexp_replace(item[80], '"', '') as item from transactions_andre_all where item[80] is not null union all
select id, regexp_replace(item[81], '"', '') as item from transactions_andre_all where item[81] is not null union all
select id, regexp_replace(item[82], '"', '') as item from transactions_andre_all where item[82] is not null union all
select id, regexp_replace(item[83], '"', '') as item from transactions_andre_all where item[83] is not null union all
select id, regexp_replace(item[84], '"', '') as item from transactions_andre_all where item[84] is not null union all
select id, regexp_replace(item[85], '"', '') as item from transactions_andre_all where item[85] is not null union all
select id, regexp_replace(item[86], '"', '') as item from transactions_andre_all where item[86] is not null union all
select id, regexp_replace(item[87], '"', '') as item from transactions_andre_all where item[87] is not null union all
select id, regexp_replace(item[88], '"', '') as item from transactions_andre_all where item[88] is not null union all
select id, regexp_replace(item[89], '"', '') as item from transactions_andre_all where item[89] is not null union all
select id, regexp_replace(item[90], '"', '') as item from transactions_andre_all where item[90] is not null union all
select id, regexp_replace(item[91], '"', '') as item from transactions_andre_all where item[91] is not null union all
select id, regexp_replace(item[92], '"', '') as item from transactions_andre_all where item[92] is not null union all
select id, regexp_replace(item[93], '"', '') as item from transactions_andre_all where item[93] is not null union all
select id, regexp_replace(item[94], '"', '') as item from transactions_andre_all where item[94] is not null union all
select id, regexp_replace(item[95], '"', '') as item from transactions_andre_all where item[95] is not null union all
select id, regexp_replace(item[96], '"', '') as item from transactions_andre_all where item[96] is not null union all
select id, regexp_replace(item[97], '"', '') as item from transactions_andre_all where item[97] is not null;

--
-- If a customer buys 2 similar items in 1 transactions, this will introduce a duplicate.
-- Those duplicates needs to be removed.
--
drop table transactions_andre_big;
create table transactions_andre_big as select distinct * from transactions_andre_temp;

--
-- drop temporary table
--
drop table transactions_andre_temp;

--
-- introduce a 'virtual item' to add additional info to the rules.
-- The virtual itme is Day of the Week.
-- 
create table transactions_andre_big_virtual as select * from transactions_andre_big union all
select id, case when order_dow = 0 then 'Monday'
           when order_dow = 1 then 'Tuesday'
           when order_dow = 2 then 'Wednessday'
           when order_dow = 3 then 'Thursday'
           when order_dow = 4 then 'Friday'
           when order_dow = 5 then 'Saturday'
           when order_dow = 6 then 'Sunday'
           else null
           end as item
from transactions_andre_all;


--
-- Create new table in order to run the Collaborative Filtering ALS algoritm
-- With this algorithm, an item->item recommendation model will be built.
--
-- Step 1: Create a table with unique product names (item), and generate a unique id
--

create table transaction_item_temp as select distinct item from transactions_andre_big;
create table transaction_item as select item, row_number over() as item_id from transaction_item_temp;
drop table transaction_item_temp
--
-- Step 2: Create a new transaction table with item_id's instead of item_names
--
-- The Collaborative Filter technique uses a 'ranking' value in order to improve recommendation
-- quality. Since in a PointOfSale situation, a ranking value is not present, we will submit the
-- value 1.0 for the ranking. Each product is equally ranked. The algorithm will still have
-- recommendation value, since some products are bought together more often then others.
-- 

create table transaction_andre_als as 
  select a.id as user_id
  ,      b.item_id as item_id
  ,      1.0 as ranking 
  from   transactions_andre_big a
  ,      transaction_item b
  where  a.item = b.item;

--
-- Step 3: create an hdfs file out of the table, as input for the algorithm.
--

INSERT OVERWRITE DIRECTORY '/user/clouderanA/andre/transactions_andre_als' ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' select * from transactions_andre_als limit 1000000;
drop table transaction_andre_als

