const {
  Document,
  Packer,
  Paragraph,
  TextRun,
  UnderlineType,
  BorderStyle,
  WidthType,
  PageBreak,
  TableOfContents,
  HeadingLevel
} = require("docx");

module.exports = (evidenceData, argument) => {
  console.log(evidenceData);
  let title = {properties: {}, children: [new Paragraph({
    children: [
      new TextRun({
        text: "DarchAI Card Document",
        bold: true,
        size: 48,
      }),
    ],
    alignment: "center",
  }),
  new Paragraph({
    children: [
      new TextRun({
        text: "Argument: " + argument,
        bold: true,
        underline: {},
        size: 36,
      }),
    ],
    alignment: "center",
  })
]}
  let ev = evidenceData.map((evidence) => ({
    properties: {},
    children: [
      new Paragraph({
        children: [
          new TextRun({
            text: evidence.tagline,
            bold: true,
            size: 30,
            color: '2460bf',
          }),
        ],
        border: {
          bottom: {
            color: 'auto',
            space: 1,
            value: 'single',
            size: 6,
            style: BorderStyle.SINGLE,
          },
        },
        spacing: {
          after: 200,
        },
      }),
      new Paragraph({
        children: [
          new TextRun({
            text: evidence.url,
            underline: {
              type: 'single',
            },
            size: 26,
            color: '24a3c9',
          }),
          new TextRun('\n'),
        ],
        spacing: {
          after: 400,
        },
      }),
      new Paragraph({
        children: evidence.relevant_sentences.flatMap(([sentence, isRelevant, previousContext, afterContext], sentenceIndex, allSentences) => {
          const children = [];

          previousContext.forEach(([context, similarity]) => {
            if (similarity > 0.5) {
              children.push(
                new TextRun({
                  text: context + ' ',
                  underline: {
                    type: 'single',
                  },
                  size: 24,
                })
              );
            } else {
              children.push(
                new TextRun({
                  text: context + ' ',
                  size: 18,
                })
              );
            }
          });

          if (isRelevant) {
            children.push(
              new TextRun({
                text: sentence,
                bold: true,
                size: 24,
              })
            );
          } else {
            children.push(
              new TextRun({
                text: sentence,
                size: 18,
              })
            );
          }

          afterContext.forEach(([context, similarity]) => {
            if (similarity > 0.5) {
              children.push(
                new TextRun({
                  text: context + ' ',
                  underline: {
                    type: 'single',
                  },
                  size: 24,
                })
              );
            } else {
              children.push(
                new TextRun({
                  text: context + ' ',
                  size: 12,
                })
              );
            }
          });

          if (sentenceIndex < allSentences.length - 1) {
            children.push(new TextRun('[...] '));
          }

          return children;
        }),
        spacing: {
          line: 320, // 1.5 line spacing (in twips)
        },
      }),
    ],
  }));
    return {
    sections: [title].concat(ev)
  }
}