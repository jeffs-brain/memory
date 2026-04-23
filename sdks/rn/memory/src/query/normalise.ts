const INVISIBLES = /\u200B|\u200C|\u200D|\uFEFF/g
const NBSP = /\u00A0/g
const WHITESPACE_RUN = /\s+/g

export const normalise = (raw: string): string => {
  if (raw === '') return ''
  let output = raw.normalize('NFC')
  output = output.replace(INVISIBLES, '')
  output = output.replace(NBSP, ' ')
  output = output.replace(WHITESPACE_RUN, ' ')
  return output.trim()
}

export const lowerToken = (token: string): string => token.toLocaleLowerCase('en')
