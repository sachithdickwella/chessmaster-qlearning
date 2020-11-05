$(() => {
    const board = ChessBoard('board1', {
        draggable: true,
        dropOffBoard: 'trash'
    })
    board.start()
});

const shot = () => html2canvas(document.querySelector("#board1")).then(canvas => call(canvas.toDataURL()));

const call = (data) => {
    const form = new FormData()
    form.append('file', data)
    $.ajax({
        url: '/movement/grab',
        method: 'POST',
        enctype: 'multipart/form-data',
        contentType: false,
        processData: false,
        cache: false,
        data: form,
        success: (msg) => {
            console.log(msg)
        }
    })
}