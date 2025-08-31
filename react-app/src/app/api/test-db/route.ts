/*import { NextResponse } from 'next/server'
import dbConnect from '@/lib/mongodb/mongoose'

export async function GET() {
  try {
    await dbConnect()
    return NextResponse.json({ message: 'MongoDB connection successful!' })
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to connect to MongoDB' },
      { status: 500 }
    )
  }
}
*/

import { NextResponse } from 'next/server'

export async function GET() {
  return NextResponse.json({ 
    message: 'API route works!',
    timestamp: new Date().toISOString()
  })
}