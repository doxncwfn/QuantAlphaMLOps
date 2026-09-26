Y = F(X)

Tai thoi diem t:

Question: Trong cac stock dang niem yet trong M_t, sap xep theo theo tieu chi A

In: - Thi truong: M_t => S_1, S_2, S_3,..., S_i - History cua cac S (60 ngay) co xuat hien (t-60, t) -
Out: - Xep hang cac co phieu trong M_t vao t + delta => Order - Ranking by? - Return + Active Status? + Time to life? +

y = f(x)

Modeling: Robust

# Dataset:

# Input:

- Historical: Feature Engineering
- Cross-Sectional

# Output: Ranking vector

- Size: n (v[i] = f(x))
  sort_score(Tuple(index, score)) -> ranking by score

# How:

## 1 Stock List at t -> Historical of that list -> Input -> Output

- Pros: Easy to implement, clear logic
- Cons: Losing the data of stocks that died earlier (blind)

## 2 Take all stocks in a PIT window

- Pros: Enough data
- Cons: Output to input mapping

# Metrics?

- Foundation:

# Approach

## Filter Definition (assure $$n_{stock} = const $$)

- To choose list of stocks at a point in time (t) -> input

#### Criteria

- Consume enough insights
- Top volume?

## **Feature Engineering and Input Definition**

#### Criteria

- Input from output mapping
- Features for generical model
- Historical data
