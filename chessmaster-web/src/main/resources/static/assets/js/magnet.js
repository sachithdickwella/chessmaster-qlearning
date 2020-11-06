$(() => {
    const board = ChessBoard('board1', {
        draggable: true,
        dropOffBoard: 'trash'
    })
    board.start()
});

const shot = () => html2canvas(document.querySelector("#board1")).then(canvas => call(canvas));

const call = (canvas) => {
    canvas.toBlob((blob) => {
        const form = new FormData()
        form.append('id', $('#board1').parent('div').attr('id'))
        form.append('file', blob, 'board-frame')
        $.ajax({
            url: '/movement/grab',
            method: 'POST',
            enctype: 'multipart/form-data',
            contentType: false,
            processData: false,
            cache: false,
            data: form,
            beforeSend: (_ /* jqXHR */) => {
                /*
                 * Show ajax loader before send the frame and show until
                 * receive a response with the next move.
                 */
            },
            error: (jqXHR) => {
                console.log(jqXHR)
            }
        })
    }, 'image/png')
}