Trước khi nhóm em tiếp tục, nhóm em xin phép được hỏi ý kiến các thầy về một số vấn đề sau đây ạ:

---

### 1. Về định nghĩa _research universe_

Theo ý kiến của các thầy, _canonical research universe_ cho project nên bao gồm:

- Tất cả các **tradable securities**;
- Tất cả các **equity securities**;
- Hay chỉ **common equity**?

Nếu lựa chọn **common equity only**, nhóm em cũng xin hỏi thêm là đối với các trường hợp như multiple share classes, ADRs, preferred shares, warrants, rights, units, ETFs/ETNs,... thì nhóm em nên xử lý như thế nào cho phù hợp ạ?

---

### 2. Về những ngày dữ liệu bị thiếu/không đáng tin cậy

Trong trường hợp những ngày source snapshot bị **corruption/truncation**, theo các thầy:

- Nên giữ lại ngày đó trong timeline nhưng đánh dấu universe state là **unknown**;
- Hay loại bỏ ngày đó khỏi universe reconstruction?

Nhóm em hiện đang nghiêng về phương án giữ lại ngày trong timeline và đánh dấu là **UNKNOWN**, để tránh nhầm lẫn ngày thiếu dữ liệu thành ngày một security thực sự inactive.

---

### 3. Về các short spells

Với những _spells_ (chuỗi xuất hiện liên tục) chỉ kéo dài 1–vài sessions, theo các thầy có nên giữ lại trong quá trình identity resolution rồi sau đó mới filter theo security type hoặc research universe, hay cho phép đặt một ngưỡng tối thiểu về số sessions ngay từ đầu ạ?

---

### 4. Về các _multi-spell_ / _long-gap cases_

Đối với các ticker xuất hiện nhiều lần hoặc có gap rất dài, các thầy có muốn nhóm em:

- Resolve từng spell về security_id riêng, sau đó mới xác định xem có phải cùng một security hay không;
- Và có nên sử dụng thêm CIK / FIGI / SEC filings để xác minh identity không ạ?

Nhóm em hiện đang nghiêng về phương án này vì ticker có thể bị reuse và CIK lại là issuer-level, không phải security-level.

---

### 5. Về exchange/calendar

Hiện tại nhóm em đang sử dụng NYSE trading calendar làm common observation calendar cho toàn bộ universe. Các thầy có thấy giả định này phù hợp với project không, hay về sau nên xem xét sử dụng trading calendar riêng cho từng exchange/security?

---

### 6. Về mục tiêu của historical universe

Cuối cùng, nhóm em muốn xác nhận lại: mục tiêu của universe reconstruction là tái tạo historical universe đúng với từng thời điểm mà source ghi nhận, hay cố gắng xây dựng một economically correct/investable universe (ví dụ chỉ giữ các securities thật sự phù hợp với strategy)?

---

Nhóm em rất mong nhận được ý kiến đóng góp của các thầy để chốt các assumption này trước khi tiến hành identity resolution và xây dựng security_master; bởi các quyết định này sẽ ảnh hưởng trực tiếp đến các bước xử lý dữ liệu và backtest về sau ạ.

Nhóm em xin chân thành cảm ơn các thầy!
